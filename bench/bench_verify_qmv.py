"""Micro-benchmark: INT4 quantized matvec at M=3 (MTP verify step).

Compares:
  A) Stock QuantizedLinear.__call__
  B) Pre-dequantize to BF16 + BF16 matmul
  C) Custom Metal INT4 kernel via mx.fast.metal_kernel()
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


def create_qmv4_kernel(group_size: int):
    # INT4 per-group scaling: each output element has K/group_size scales,
    # one per group of group_size input elements.
    # scales shape: (N, K/group_size) flattened as (N * K/group_size,)
    # w shape: (N, K/8) uint32 packed INT4
    # x shape: (M, K) bf16
    # y shape: (M, N) float32
    source = f"""
    uint tid = thread_position_in_grid.x;
    uint n = tid / 3;
    uint m = tid % 3;
    if (n >= N_size) return;

    // Row byte offset: each row is K_size/2 bytes (INT4: 2 values per byte)
    const device uint8_t* w_bytes = (const device uint8_t*)w;
    const device uint16_t* w_row = (const device uint16_t*)(w_bytes + n * (K_size / 2));

    uint n_groups = K_size / {group_size};
    uint vals_per_group = {group_size};  // INT4 values per group
    uint iters_per_group = vals_per_group / 4;  // uint16 iters per group (4 vals each)

    // scales and biases are (N, n_groups) flattened row-major
    uint scale_row = n * n_groups;

    float accum = 0.0f;
    for (uint g = 0; g < n_groups; g++) {{
        float scale = scales[scale_row + g];
        float bias_val = (biases) ? biases[scale_row + g] : 0.0f;
        float group_accum = 0.0f;
        float group_x_sum = 0.0f;
        uint base = g * iters_per_group;
        for (uint i = 0; i < iters_per_group; i++) {{
            uint16_t packed = w_row[base + i];
            uint k = (base + i) * 4;
            float x0 = x[m * K_size + k];
            float x1 = x[m * K_size + k + 1];
            float x2 = x[m * K_size + k + 2];
            float x3 = x[m * K_size + k + 3];
            group_accum += x0 * float(packed & 0x000fu);
            group_accum += x1 * float((packed >> 4) & 0x000fu);
            group_accum += x2 * float((packed >> 8) & 0x000fu);
            group_accum += x3 * float((packed >> 12) & 0x000fu);
            group_x_sum += x0 + x1 + x2 + x3;
        }}
        accum += scale * group_accum + bias_val * group_x_sum;
    }}

    y[m * N_size + n] = static_cast<float>(accum);
    """
    return mx.fast.metal_kernel(
        name=f"qmv4_m3_gs{group_size}",
        input_names=["x", "w", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )


def main():
    parser = argparse.ArgumentParser(description="verify_qmv micro-benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    parser.add_argument("--sidecar", default="mlx-community/Qwen3.5-9B-MTP-4bit")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    n_runs = args.n_runs
    warmup = args.warmup

    # --- load model + inject MTP ---
    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)
    inject_mtp_support(model, mtp_sidecar=args.sidecar)
    validate_mtp_support(model)
    if hasattr(model, "language_model"):
        model = model.language_model

    # --- extract QuantizedLinear from MTP head ---
    # MTP structure: model.mtp layers are _MTPDecoderLayer with .mlp
    # model.mtp._layers[0].mlp or model.mtp.layers[0].mlp
    mtp_layers = model.mtp.layers
    mlp = mtp_layers[0].mlp
    if hasattr(mlp, "gate_proj"):
        q_linear = mlp.gate_proj
        layer_path = "mtp.layers[0].mlp.gate_proj"
    elif hasattr(mlp, "experts"):
        q_linear = mlp.experts[0].gate_proj
        layer_path = "mtp.layers[0].mlp.experts[0].gate_proj"
    else:
        # Try gate_proj on the module itself
        q_linear = mlp
        layer_path = "mtp.layers[0].mlp"

    N = q_linear.weight.shape[0]
    K = q_linear.weight.shape[1] * (32 // q_linear.bits)  # INT4: 8 values per uint32
    group_size = q_linear.group_size
    bits = q_linear.bits

    print(f"\n=== verify_qmv Micro-Benchmark (M=3) ===")
    print(f"Layer: {layer_path}")
    print(f"  Shape: N={N}, K={K}, group_size={group_size}, bits={bits}")
    print(f"  MTP hidden_size: {K}")

    x = mx.random.normal((3, K)).astype(mx.bfloat16)
    mx.eval(x)

    # --- A: Stock QuantizedLinear ---
    for _ in range(warmup):
        y = q_linear(x)
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        y = q_linear(x)
        mx.eval(y)
    mx.synchronize(None)
    stock_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\nApproach A — Stock QuantizedLinear:")
    print(f"  {stock_ms:.2f} ms/call ({n_runs} runs)")

    # --- B: Pre-dequantized BF16 matmul ---
    dense = mx.dequantize(
        q_linear.weight,
        q_linear.scales,
        q_linear.biases,
        group_size=q_linear.group_size,
        bits=q_linear.bits,
        mode=getattr(q_linear, "mode", "affine"),
    ).astype(mx.bfloat16)
    mx.eval(dense)

    for _ in range(warmup):
        y = x @ dense.T
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        y = x @ dense.T
        mx.eval(y)
    mx.synchronize(None)
    dequant_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\nApproach B — Pre-dequantized BF16 matmul:")
    print(
        f"  {dequant_ms:.2f} ms/call ({n_runs} runs)"
        f"  \u2192 {stock_ms / dequant_ms:.2f}x vs stock"
    )

    # --- C: Custom Metal INT4 kernel ---
    metal_ok = False
    try:
        biases = q_linear.biases
        if biases is None:
            biases = mx.zeros((N // group_size,), dtype=mx.float32)
    except AttributeError:
        biases = mx.zeros((N // group_size,), dtype=mx.float32)

    try:
        kernel = create_qmv4_kernel(group_size)

        w_int = q_linear.weight
        scales = q_linear.scales
        K_arr = mx.array(K, dtype=mx.uint32)
        N_arr = mx.array(N, dtype=mx.uint32)

        for _ in range(warmup):
            y = kernel(
                inputs=[x, w_int, scales, biases, K_arr, N_arr],
                output_shapes=[(3, N)],
                output_dtypes=[mx.float32],
                grid=(3 * N, 1, 1),
                threadgroup=(min(256, 3 * N), 1, 1),
            )
            mx.eval(y)

        # --- Correctness check ---
        y_kernel = kernel(
            inputs=[x, w_int, scales, biases, K_arr, N_arr],
            output_shapes=[(3, N)],
            output_dtypes=[mx.float32],
            grid=(3 * N, 1, 1),
            threadgroup=(min(256, 3 * N), 1, 1),
        )
        # mx.fast.metal_kernel returns a list of arrays
        if isinstance(y_kernel, list):
            y_kernel = y_kernel[0]
        y_stock = q_linear(x)  # stock output (dequant + matmul)
        # y_stock is from QuantizedLinear which may be bf16; cast to float32
        if y_stock.dtype != mx.float32:
            y_stock = y_stock.astype(mx.float32)
        mx.eval(y_kernel, y_stock)
        max_err = float(mx.abs(y_kernel - y_stock).max())
        mean_err = float(mx.abs(y_kernel - y_stock).mean())
        print(f"\nCorrectness check (kernel vs stock):")
        print(f"  Max absolute error:  {max_err:.6f}")
        print(f"  Mean absolute error: {mean_err:.6f}")
        if max_err > 0.1:
            print(f"  ⚠ Large error — kernel output is likely WRONG")
        elif max_err > 0.01:
            print(f"  ⚠ Moderate error — INT4 rounding, acceptable")
        else:
            print(f"  ✓ Kernel output matches stock")

        t0 = time.perf_counter()
        for _ in range(n_runs):
            y = kernel(
                inputs=[x, w_int, scales, biases, K_arr, N_arr],
                output_shapes=[(3, N)],
                output_dtypes=[mx.float32],
                grid=(3 * N, 1, 1),
                threadgroup=(min(256, 3 * N), 1, 1),
            )
            mx.eval(y)
        mx.synchronize(None)
        metal_ms = (time.perf_counter() - t0) / n_runs * 1000
        metal_ok = True

    except (AttributeError, Exception) as exc:
        print(f"\nApproach C — Custom Metal INT4 kernel: SKIPPED ({exc})")

    if metal_ok:
        print(f"\nApproach C — Custom Metal INT4 kernel:")
        print(
            f"  {metal_ms:.2f} ms/call ({n_runs} runs)"
            f"  \u2192 {stock_ms / metal_ms:.2f}x vs stock"
        )


if __name__ == "__main__":
    main()
