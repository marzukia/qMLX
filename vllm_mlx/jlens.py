# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx jlens`` — read a model's internal "draft" with the Jacobian lens.

The Jacobian lens (J-lens) linearly transports a residual-stream vector at an
intermediate layer into the final-layer basis and decodes it with the model's
own unembedding, revealing what an internal activation is *disposed* to make the
model say — before it is said. Concretely, for layer ``l`` we estimate the
averaged transport ``J_l = E[d h_final / d h_l]`` over a small text corpus and
read ``unembed(J_l . h)``. Unlike the plain logit lens, J-lens also accounts for
the influence an activation has on *future* tokens, so it surfaces intermediate
reasoning steps and concept clusters that the logit lens reads as noise.

``J_l . h`` is computed with a Jacobian–vector product (forward-mode autodiff),
so the full ``d x d`` Jacobian is never materialised and the cost is
``O(corpus_size)`` forward passes per layer. JVP flows through quantised
(4-bit/8-bit) weights, so this runs directly on the cached MLX models with a
finite-difference fallback for architectures where autodiff is unavailable.

This is a read-only interpretability command; it loads weights but never serves.
"""

from __future__ import annotations

import json
import os
import sys

# A small, generic web-text corpus used to average the transport J_l. Kept
# short and topically diverse so the estimate is cheap yet stable.
_CORPUS = [
    "The weather today is quite nice and sunny.",
    "Machine learning models are trained on large datasets.",
    "He drove his car along the coastal highway at dawn.",
    "Scientists discovered a new species deep in the ocean.",
    "Water boils at one hundred degrees under normal pressure.",
    "The committee agreed to postpone the final decision.",
    "A gentle breeze moved through the tall green trees.",
    "The recipe calls for two cups of flour and one egg.",
    "The train departed from the station exactly on time.",
    "Investors are watching interest rates very closely now.",
    "Light from the sun travels through empty space quickly.",
    "The artist painted a portrait using bright warm colors.",
]


class UnsupportedArchitectureError(Exception):
    """Raised when a model's architecture is not (yet) J-lens compatible."""


def _load_model(model_path: str):
    """Load an mlx_lm model, installing the MLX hardware-compat shim first."""
    # The shim must run before ``from mlx_lm import load`` (see cli.py / #404).
    from . import _mlx_compat

    _mlx_compat.install()
    from mlx_lm import load

    return load(model_path)


def _get_inner(model):
    """Locate the text transformer holding .embed_tokens / .layers / .norm."""
    for path in ("model", "language_model.model", "language_model", "transformer"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
            return obj
    raise UnsupportedArchitectureError(
        "could not locate a standard residual-stream text model "
        "(embed_tokens / layers / norm)"
    )


def _get_unembed(model, inner):
    """Return a callable mapping a normed hidden vector to vocab logits."""
    for path in ("lm_head", "language_model.lm_head"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and callable(obj):
            return obj
    return inner.embed_tokens.as_linear


def _is_content(tok_str: str) -> bool:
    """True for a meaningful vocabulary token (not padding / punctuation)."""
    return (
        bool(tok_str)
        and not tok_str.startswith("_")
        and len(tok_str) > 1
        and any(c.isalnum() for c in tok_str)
    )


class JLensAnalyzer:
    """Compute corpus-averaged J-lens readouts for a loaded MLX model."""

    def __init__(self, model, tokenizer):
        import mlx.core as mx

        self._mx = mx
        self.model = model
        self.tok = tokenizer
        self.inner = _get_inner(model)
        self.unembed = _get_unembed(model, self.inner)
        self.n_layers = len(self.inner.layers)
        self._use_jvp = True
        # Decoder blocks differ in call signature: some are layer(x, mask) with
        # a defaulted cache, others require layer(x, mask, cache) explicitly
        # (e.g. mlx_lm StableLM). Probed once on first use, then reused.
        self._layer_mode = None
        try:
            from mlx_lm.models.base import create_attention_mask

            self._mk_mask = create_attention_mask
        except Exception:  # pragma: no cover - depends on mlx_lm version
            self._mk_mask = None
        self._corpus = [self._pre_states(t) for t in _CORPUS]

    # -- forward / transport helpers ------------------------------------
    def _mask(self, h):
        if self._mk_mask is not None:
            return self._mk_mask(h, None)
        mx = self._mx
        length = h.shape[1]
        return mx.triu(mx.full((length, length), -1e9, dtype=h.dtype), k=1)

    def _run_layer(self, layer, h, mask):
        """Call a decoder block, tolerating both cache-optional and
        cache-required signatures. The working form is detected once."""
        if self._layer_mode == "no_cache":
            return layer(h, mask)
        if self._layer_mode == "cache":
            return layer(h, mask, None)
        try:  # probe: cache-optional signature layer(x, mask)
            out = layer(h, mask)
            self._layer_mode = "no_cache"
            return out
        except TypeError:  # cache-required signature layer(x, mask, cache)
            out = layer(h, mask, None)
            self._layer_mode = "cache"
            return out

    def _pre_states(self, text):
        """Return (pre, mask, seq_len) where pre[l] enters layer l (pre[0]=embed)."""
        mx = self._mx
        ids = mx.array(self.tok.encode(text))[None]
        try:
            h = self.inner.embed_tokens(ids)
            mask = self._mask(h)
            pre = []
            for layer in self.inner.layers:
                pre.append(h)
                h = self._run_layer(layer, h, mask)
        except (TypeError, ValueError) as exc:  # linear-attn / hybrid masks
            raise UnsupportedArchitectureError(
                f"forward pass is not J-lens compatible ({type(exc).__name__}: {exc})"
            ) from exc
        return pre, mask, ids.shape[1]

    def _transport(self, pre_l, mask, layer_idx, tangent):
        """Return (d h_final / d h_l) . tangent as a full-sequence array."""
        mx = self._mx
        layers = self.inner.layers

        def f(x):
            h = x
            for layer in layers[layer_idx:]:
                h = self._run_layer(layer, h, mask)
            return self.inner.norm(h)

        if self._use_jvp:
            try:
                # mx.jvp returns (outputs, jvps) as lists, one entry per fn
                # output; f returns a single [batch, seq, hidden] array, so we
                # unwrap the length-1 list. jout[0] keeps the batch dim (it is
                # the full 3-D array), matching the finite-difference path below.
                _, jout = mx.jvp(f, (pre_l,), (tangent,))
                return jout[0] if isinstance(jout, (list, tuple)) else jout
            except Exception:  # autodiff unsupported → forward-mode fallback
                self._use_jvp = False
        eps = 1e-3
        return (f(pre_l + eps * tangent) - f(pre_l)) / eps

    def _jlens(self, h_test, layer_idx):
        """Estimate ( E[d h_final / d h_l] . h_test ) as vocab logits."""
        mx = self._mx
        acc = mx.zeros_like(h_test)
        count = 0
        for pre, mask, length in self._corpus:
            if length < 3:
                continue
            src = length // 2
            tangent = mx.zeros_like(pre[layer_idx])
            tangent[:, src, :] = h_test[0]
            out = self._transport(pre[layer_idx], mask, layer_idx, tangent)
            acc = acc + out[:, src:, :].mean(axis=1)  # average over t >= src
            count += 1
        return self.unembed(acc / max(1, count))[0].astype(mx.float32)

    def _logit_lens(self, h):
        return self.unembed(self.inner.norm(h))[0].astype(self._mx.float32)

    def _top(self, vec, k=6):
        idx = self._mx.argsort(-vec)[:k]
        return [self.tok.decode([int(i)]).strip() for i in idx.tolist()]

    # -- public API -----------------------------------------------------
    def _complete(self, prompt, max_tokens=6):
        """A short greedy completion, for context next to the J-lens answer."""
        try:
            from mlx_lm import generate

            return generate(
                self.model,
                self.tok,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            ).strip()
        except Exception:  # pragma: no cover - generation is best-effort context
            return ""

    def analyze(self, prompt, step=2):
        mx = self._mx
        completion = self._complete(prompt)
        pre, _mask, _len = self._pre_states(prompt)
        layers = list(range(0, self.n_layers, step))
        if layers[-1] != self.n_layers - 1:
            layers.append(self.n_layers - 1)
        jl_top, ll_top, jl_vecs = {}, {}, {}
        for li in layers:
            h = pre[li][:, -1, :]
            jl_vec = self._jlens(h, li)
            jl_vecs[li] = jl_vec
            jl_top[li] = self._top(jl_vec)
            ll_top[li] = self._top(self._logit_lens(h))
        # Anchor the answer to the final-layer argmax token id so its rank can
        # be traced across layers (the Fig-5 "rank trajectory" view).
        answer_id = int(mx.argmax(jl_vecs[layers[-1]]))
        answer = self.tok.decode([answer_id]).strip()
        answer_rank = {
            li: int((jl_vecs[li] > jl_vecs[li][answer_id]).sum()) for li in layers
        }

        def first(topd):
            for li in layers:
                if answer in topd[li]:
                    return li
            return None

        return {
            "prompt": prompt,
            "completion": completion,
            "n_layers": self.n_layers,
            "layers": layers,
            "answer": answer,
            "answer_rank_by_layer": answer_rank,
            "jlens_first_layer": first(jl_top),
            "logit_lens_first_layer": first(ll_top),
            "jlens_by_layer": jl_top,
            "logit_lens_by_layer": ll_top,
            "transport": "jvp" if self._use_jvp else "finite-diff",
        }


def _workspace_signal(result):
    """Cheap, experimental heuristic in [0, 1]: mid-layer concept density plus
    how many layers earlier J-lens locks the answer vs the logit lens.

    NOTE: this is a rough signal, not a validated capability metric — it partly
    reflects answer *confidence*, not only reasoning structure. A rigorous
    version would calibrate against a J-space ablation on multi-step tasks.
    """
    n = result["n_layers"]
    layers = result["layers"]
    jl = result["jlens_by_layer"]
    mids = [li for li in layers if n // 4 <= li < 3 * n // 4]
    total = sum(len(jl[li]) for li in mids) or 1
    density = sum(_is_content(t) for li in mids for t in jl[li]) / total
    jlf, llf = result["jlens_first_layer"], result["logit_lens_first_layer"]
    lead = (llf - jlf) / n if (jlf is not None and llf is not None) else 0.0
    return round(min(1.0, 0.5 * density + max(0.0, lead)), 2), round(density, 3)


def render_text(result, model_label):
    n = result["n_layers"]
    layers = result["layers"]
    jl = result["jlens_by_layer"]
    out = []
    out.append("")
    out.append(f"rapid-mlx jlens · {model_label} · {result['prompt']!r}")
    if result.get("completion"):
        out.append(f"  model continues → {result['completion']!r}")
    out.append(f"  [{result['transport']}]  {n} layers")
    out.append("")
    out.append(
        "internal trajectory  (J-lens: what the model is disposed to say, by depth)"
    )
    bands = [
        ("early", 0, n // 4),
        ("mid  ", n // 4, n // 2),
        ("deep ", n // 2, 3 * n // 4),
        ("final", 3 * n // 4, n),
    ]
    for name, lo, hi in bands:
        seen, concepts = set(), []
        for li in layers:
            if lo <= li < hi:
                for t in jl[li]:
                    if _is_content(t) and t.lower() not in seen:
                        seen.add(t.lower())
                        concepts.append(t)
        tag = "  ".join(concepts[:6]) if concepts else "--"
        out.append(f"  {name}  L{lo:>2}-{hi - 1:<2}  ·  {tag}")

    ans = result["answer"]
    jlf = result["jlens_first_layer"]
    llf = result["logit_lens_first_layer"]
    out.append("")
    out.append(f"answer {ans!r}:")
    if jlf is not None:
        pct = round(100 * jlf / n)
        out.append(
            f"  crystallizes at  L{jlf}/{n}  ({pct}% depth)  →  early-exit headroom ~{100 - pct}%"
        )
    if jlf is not None and llf is not None:
        out.append(
            f"  J-lens lead over logit lens:  +{llf - jlf} layers  (logit lens locks at L{llf})"
        )

    score, density = _workspace_signal(result)
    filled = int(score * 10)
    bar = "#" * filled + "." * (10 - filled)
    out.append("")
    out.append(f"workspace signal  {score}  {bar}   (experimental heuristic)")
    lead_str = (llf - jlf) if (jlf is not None and llf is not None) else "n/a"
    out.append(
        f"  mid-layer concept density {round(density * 100)}%  ·  J-lens lead {lead_str}L"
    )
    out.append("")
    return "\n".join(out)


def render_verbose(result, model_label):
    """Fuller "Fig-5"-style view: the concise summary, then per-layer ranked
    readouts and the answer token's rank trajectory across depth."""
    layers = result["layers"]
    jl = result["jlens_by_layer"]
    ll = result["logit_lens_by_layer"]
    ranks = result.get("answer_rank_by_layer", {})
    ans = result["answer"]
    out = [render_text(result, model_label).rstrip("\n"), ""]
    out.append("per-layer readouts  (top tokens, ranked left→right)")
    out.append(f"{'L':>4} | {'J-lens':<46} | logit lens")
    for li in layers:
        out.append(f"{li:>4} | {'  '.join(jl[li]):<46} | {'  '.join(ll[li])}")
    if ranks:
        out.append("")
        out.append(f"rank trajectory of answer {ans!r}  (0 = top-1, lower is stronger)")
        out.append("  " + "  ".join(f"L{li}:{ranks[li]}" for li in layers))
    out.append("")
    return "\n".join(out)


def jlens_command(args):
    """Entry point for ``rapid-mlx jlens``."""
    if getattr(args, "step", 1) < 1:
        print(f"\n  Error: --step must be a positive integer (got {args.step}).\n")
        sys.exit(2)
    model_path = args.model
    label = getattr(args, "_original_alias", None) or os.path.basename(model_path)
    try:
        model, tokenizer = _load_model(model_path)
        analyzer = JLensAnalyzer(model, tokenizer)
        result = analyzer.analyze(args.prompt, step=args.step)
    except UnsupportedArchitectureError as exc:
        print(f"\n  J-lens does not support this model architecture yet: {exc}")
        print("  Supported: standard dense decoder transformers with full attention")
        print(
            "  (e.g. Qwen3, Llama, Phi). Linear-attention / hybrid / VLM models are not yet handled.\n"
        )
        sys.exit(2)

    if getattr(args, "json", False):
        payload = {k: v for k, v in result.items()}
        payload["model"] = model_path
        payload["workspace_signal"], payload["mid_layer_density"] = _workspace_signal(
            result
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif getattr(args, "verbose", False):
        print(render_verbose(result, label))
    else:
        print(render_text(result, label))
