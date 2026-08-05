"""Fused MLP kernel benchmark for MTP head.

Compares stock MLP (3 separate QuantizedLinear calls + SiLU + mul)
against a fused Metal kernel that:
  1. Fuses gate_proj + up_proj + SiLU + element-wise multiply (Kernel 1)
  2. Runs down_proj as a separate kernel (Kernel 2)

Key insight: Kernel 1 reads x once for both gate and up projections,
eliminating the redundant input read and reducing kernel launch overhead.

MLP structure (SwiGLU):
  intermediate = SiLU(gate_proj(x)) * up_proj(x)
  output = down_proj(intermediate)
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
    inject_mtp_support,
    validate_mtp_support,
)


# =========================================================================
# Kernel 1: fused gate + up + SiLU + element-wise multiply
# Each thread handles one (m, n) pair. Reads x[m, :] once, computes
# both gate[n] and up[n] via INT4 per-group matvec, applies SiLU(gate)*up.
# =========================================================================
def create_fused_gate_up_kernel(group_size: int):
    source = f"""
    uint tid = thread_position_in_grid.x;
    uint n = tid / 3;       // output element (0..N-1)
    uint m = tid % 3;       // which of the 3 M vectors
    if (n >= N_size) return;

    // Row byte offsets for gate and up weights
    const device uint8_t* gate_bytes = (const device uint8_t*)gate_w;
    const device uint16_t* gate_row = (const device uint16_t*)(gate_bytes + n * (K_size / 2));
    const device uint8_t* up_bytes = (const device uint8_t*)up_w;
    const device uint16_t* up_row = (const device uint16_t*)(up_bytes + n * (K_size / 2));

    uint n_groups = K_size / {group_size};
    uint iters_per_group = {group_size} / 4;

    float gate_val = 0.0f;
    float up_val = 0.0f;

    for (uint g = 0; g < n_groups; g++) {{
        uint scale_idx = n * n_groups + g;
        float gate_scale = gate_scales[scale_idx];
        float gate_bias = (gate_biases) ? gate_biases[scale_idx] : 0.0f;
        float up_scale = up_scales[scale_idx];
        float up_bias = (up_biases) ? up_biases[scale_idx] : 0.0f;

        float gate_group = 0.0f;
        float up_group = 0.0f;
        float x_sum = 0.0f;
        uint base = g * iters_per_group;

        for (uint i = 0; i < iters_per_group; i++) {{
            uint k = (base + i) * 4;
            float x0 = x[m * K_size + k];
            float x1 = x[m * K_size + k + 1];
            float x2 = x[m * K_size + k + 2];
            float x3 = x[m * K_size + k + 3];
            x_sum += x0 + x1 + x2 + x3;

            uint16_t g_packed = gate_row[base + i];
            gate_group += x0 * float(g_packed & 0x000fu);
            gate_group += x1 * float((g_packed >> 4) & 0x000fu);
            gate_group += x2 * float((g_packed >> 8) & 0x000fu);
            gate_group += x3 * float((g_packed >> 12) & 0x000fu);

            uint16_t u_packed = up_row[base + i];
            up_group += x0 * float(u_packed & 0x000fu);
            up_group += x1 * float((u_packed >> 4) & 0x000fu);
            up_group += x2 * float((u_packed >> 8) & 0x000fu);
            up_group += x3 * float((u_packed >> 12) & 0x000fu);
        }}

        gate_val += gate_scale * gate_group + gate_bias * x_sum;
        up_val += up_scale * up_group + up_bias * x_sum;
    }}

    // SiLU(gate) * up
    float sigmoid = 1.0f / (1.0f + exp(-gate_val));
    float activated = gate_val * sigmoid * up_val;

    out[m * N_size + n] = activated;
    """
    return mx.fast.metal_kernel(
        name=f"fused_gate_up_gs{group_size}",
        input_names=[
            "x", "gate_w", "gate_scales", "gate_biases",
            "up_w", "up_scales", "up_biases",
            "K_size", "N_size",
        ],
        output_names=["out"],
        source=source,
    )


# =========================================================================
# Kernel 2: down projection (single INT4 matvec, reuse verified pattern)
# Each thread handles one (m, k_out) pair.
# =========================================================================
def create_down_proj_kernel(group_size: int):
    source = f"""
    uint tid = thread_position_in_grid.x;
    uint k_out = tid / 3;   // output element (0..K_out-1)
    uint m = tid % 3;       // which of the 3 M vectors
    if (k_out >= K_out_size) return;

    const device uint8_t* w_bytes = (const device uint8_t*)down_w;
    const device uint16_t* w_row = (const device uint16_t*)(w_bytes + k_out * (N_size / 2));

    uint n_groups = N_size / {group_size};
    uint iters_per_group = {group_size} / 4;

    float accum = 0.0f;
    for (uint g = 0; g < n_groups; g++) {{
        uint scale_idx = k_out * n_groups + g;
        float scale = down_scales[scale_idx];
        float bias_val = (down_biases) ? down_biases[scale_idx] : 0.0f;
        float group_accum = 0.0f;
        float x_sum = 0.0f;
        uint base = g * iters_per_group;

        for (uint i = 0; i < iters_per_group; i++) {{
            uint n = (base + i) * 4;
            float x0 = intermediate[m * N_size + n];
            float x1 = intermediate[m * N_size + n + 1];
            float x2 = intermediate[m * N_size + n + 2];
            float x3 = intermediate[m * N_size + n + 3];
            x_sum += x0 + x1 + x2 + x3;

            uint16_t packed = w_row[base + i];
            group_accum += x0 * float(packed & 0x000fu);
            group_accum += x1 * float((packed >> 4) & 0x000fu);
            group_accum += x2 * float((packed >> 8) & 0x000fu);
            group_accum += x3 * float((packed >> 12) & 0x000fu);
        }}

        accum += scale * group_accum + bias_val * x_sum;
    }}

    out[m * K_out_size + k_out] = accum;
    """
    return mx.fast.metal_kernel(
        name=f"down_proj_gs{group_size}",
        input_names=[
            "intermediate", "down_w", "down_scales", "down_biases",
            "N_size", "K_out_size",
        ],
        output_names=["out"],
        source=source,
    )


def main():
    parser = argparse.ArgumentParser(description="Fused MLP kernel benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    parser.add_argument("--sidecar", default="mlx-community/Qwen3.5-9B-MTP-4bit")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    print("Loading model ...")
    model, tokenizer = load(args.model)
    inject_mtp_support(model, mtp_sidecar=args.sidecar)
    validate_mtp_support(model)
    if hasattr(model, "language_model"):
        model = model.language_model

    mlp = model.mtp.layers[0].mlp
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj

    K = gate.weight.shape[1] * (32 // gate.bits)
    N = gate.weight.shape[0]  # intermediate_size
    K_out = down.weight.shape[0]  # hidden_size
    gs = gate.group_size

    print(f"\n=== Fused MLP Benchmark (M=3) ===")
    print(f"gate_proj: ({N}, {K}) INT{gate.bits}, gs={gs}")
    print(f"up_proj:   ({N}, {K}) INT{up.bits}")
    print(f"down_proj: ({K_out}, {N}) INT{down.bits}")

    x = mx.random.normal((3, K)).astype(mx.bfloat16)
    mx.eval(x)

    # --- Stock MLP forward ---
    def stock_mlp(x):
        gate_out = gate(x)
        up_out = up(x)
        activated = nn.silu(gate_out) * up_out
        return down(activated)

    for _ in range(args.warmup):
        y = stock_mlp(x)
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(args.n_runs):
        y = stock_mlp(x)
        mx.eval(y)
    mx.synchronize(None)
    stock_ms = (time.perf_counter() - t0) / args.n_runs * 1000

    print(f"\nStock MLP (3x QuantizedLinear + SiLU + mul):")
    print(f"  {stock_ms:.2f} ms/call ({args.n_runs} runs)")

    # --- Fused MLP ---
    fused_ok = False
    try:
        # Biases
        gate_biases = gate.biases
        up_biases = up.biases
        down_biases = down.biases
        if gate_biases is None:
            gate_biases = mx.zeros((N, K // gs), dtype=mx.float32)
        if up_biases is None:
            up_biases = mx.zeros((N, K // gs), dtype=mx.float32)
        if down_biases is None:
            down_biases = mx.zeros((K_out, N // down.group_size), dtype=mx.float32)

        gate_kernel = create_fused_gate_up_kernel(gs)
        down_kernel = create_down_proj_kernel(down.group_size)

        K_arr = mx.array(K, dtype=mx.uint32)
        N_arr = mx.array(N, dtype=mx.uint32)
        K_out_arr = mx.array(K_out, dtype=mx.uint32)

        # Warmup
        for _ in range(args.warmup):
            intermediate = gate_kernel(
                inputs=[x, gate.weight, gate.scales, gate_biases,
                         up.weight, up.scales, up_biases, K_arr, N_arr],
                output_shapes=[(3, N)],
                output_dtypes=[mx.float32],
                grid=(3 * N, 1, 1),
                threadgroup=(min(256, 3 * N), 1, 1),
            )
            if isinstance(intermediate, list):
                intermediate = intermediate[0]
            y_fused = down_kernel(
                inputs=[intermediate, down.weight, down.scales, down_biases,
                         N_arr, K_out_arr],
                output_shapes=[(3, K_out)],
                output_dtypes=[mx.float32],
                grid=(3 * K_out, 1, 1),
                threadgroup=(min(256, 3 * K_out), 1, 1),
            )
            if isinstance(y_fused, list):
                y_fused = y_fused[0]
            mx.eval(y_fused)

        # Correctness check
        y_stock = stock_mlp(x)
        if y_stock.dtype != mx.float32:
            y_stock = y_stock.astype(mx.float32)
        intermediate = gate_kernel(
            inputs=[x, gate.weight, gate.scales, gate_biases,
                     up.weight, up.scales, up_biases, K_arr, N_arr],
            output_shapes=[(3, N)],
            output_dtypes=[mx.float32],
            grid=(3 * N, 1, 1),
            threadgroup=(min(256, 3 * N), 1, 1),
        )
        if isinstance(intermediate, list):
            intermediate = intermediate[0]
        y_fused = down_kernel(
            inputs=[intermediate, down.weight, down.scales, down_biases,
                     N_arr, K_out_arr],
            output_shapes=[(3, K_out)],
            output_dtypes=[mx.float32],
            grid=(3 * K_out, 1, 1),
            threadgroup=(min(256, 3 * K_out), 1, 1),
        )
        if isinstance(y_fused, list):
            y_fused = y_fused[0]
        mx.eval(y_fused, y_stock)

        max_err = float(mx.abs(y_fused - y_stock).max())
        mean_err = float(mx.abs(y_fused - y_stock).mean())
        print(f"\nCorrectness (fused vs stock):")
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        if max_err > 0.1:
            print(f"  ⚠ Large error — kernel is WRONG")
        else:
            print(f"  ✓ Fused output matches stock")

        # Timing
        for _ in range(args.n_runs):
            intermediate = gate_kernel(
                inputs=[x, gate.weight, gate.scales, gate_biases,
                         up.weight, up.scales, up_biases, K_arr, N_arr],
                output_shapes=[(3, N)],
                output_dtypes=[mx.float32],
                grid=(3 * N, 1, 1),
                threadgroup=(min(256, 3 * N), 1, 1),
            )
            if isinstance(intermediate, list):
                intermediate = intermediate[0]
            y_fused = down_kernel(
                inputs=[intermediate, down.weight, down.scales, down_biases,
                         N_arr, K_out_arr],
                output_shapes=[(3, K_out)],
                output_dtypes=[mx.float32],
                grid=(3 * K_out, 1, 1),
                threadgroup=(min(256, 3 * K_out), 1, 1),
            )
            if isinstance(y_fused, list):
                y_fused = y_fused[0]
            mx.eval(y_fused)
        mx.synchronize(None)
        fused_ms = (time.perf_counter() - t0) / args.n_runs * 1000

        fused_ok = True
        print(f"\nFused MLP (2 kernels: gate+up+SiLU+mul, down):")
        print(f"  {fused_ms:.2f} ms/call ({args.n_runs} runs)")
        print(f"  Speedup: {stock_ms / fused_ms:.2f}x vs stock")

    except Exception as exc:
        print(f"\nFused MLP: FAILED — {exc}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Summary")
    print(f"{'='*50}")
    print(f"  Stock MLP:    {stock_ms:.2f} ms")
    if fused_ok:
        print(f"  Fused MLP:    {fused_ms:.2f} ms")
        print(f"  Speedup:      {stock_ms / fused_ms:.2f}x")
    else:
        print(f"  Fused MLP:    FAILED")


if __name__ == "__main__":
    main()
