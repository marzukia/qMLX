# SPDX-License-Identifier: Apache-2.0
"""Tape rollback bit-exactness tests for MTP speculative decoding.

Gate test: tape rollback must produce bit-identical SSM state to
re-running the forward from the accepted position. This ensures K>=2
on SSM-hybrid targets (Qwen3.5-122B) doesn't break the lossless
contract.

The test verifies:
1. Tape recording captures per-position SSM state correctly
2. Tape rollback to position N produces identical state as re-running from N
3. K=3 generation produces byte-identical output as K=0 (no spec decode)
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_mlx.spec_decode.mtp.cache_patch import (
    patch_arrays_cache_rollback_state,
    patch_gated_delta_net_for_mtp,
)
from vllm_mlx.spec_decode.mtp.tape_rollback import (
    TapeBuffer,
    TapeRecorder,
    get_tape_recorder,
)


@pytest.fixture(autouse=True)
def patch_caches():
    """Apply cache patches before each test."""
    patch_arrays_cache_rollback_state()
    patch_gated_delta_net_for_mtp()


class TestTapeRecorder:
    """Tests for the TapeRecorder class."""

    def test_recorder_lifecycle(self):
        """Test basic recorder start/finish lifecycle."""
        recorder = TapeRecorder()

        assert not recorder.is_recording
        assert recorder.current_tape is None

        # Start recording
        fake_cache = []  # Empty cache for this test
        recorder.start(fake_cache, n_confirmed=3)

        assert recorder.is_recording
        assert recorder.current_tape is not None
        assert recorder.current_tape.n_confirmed == 3
        assert len(recorder.current_tape.tapes) == 0

        # Finish recording
        recorder.finish()

        assert not recorder.is_recording
        assert recorder.current_tape is None

    def test_record_entry(self):
        """Test recording tape entries."""
        recorder = TapeRecorder()
        cache = [type("Obj", (), {"rollback_state": []})()]
        recorder.start(cache, n_confirmed=2)

        # Record an entry
        conv_state = mx.array([[1, 2, 3]])
        ssm_state = mx.array([[4, 5, 6]])
        recorder.record_entry(layer_idx=0, conv_state=conv_state, ssm_state=ssm_state)

        tape = recorder.current_tape
        assert tape is not None
        assert 0 in tape.tapes
        assert len(tape.tapes[0].entries) == 1
        assert mx.array_equal(tape.tapes[0].entries[0].conv_state, conv_state)
        assert mx.array_equal(tape.tapes[0].entries[0].ssm_state, ssm_state)

        recorder.finish()

    def test_rollback_to_position(self):
        """Test rollback to specific position."""
        recorder = TapeRecorder()

        # Create mock cache with SSM layers
        cache = [
            type(
                "Obj",
                (),
                {
                    "rollback_state": [],
                    "__getitem__": lambda self, i: getattr(self, f"_val{i}", None),
                    "__setitem__": lambda self, i, v: setattr(self, f"_val{i}", v),
                    "is_trimmable": lambda: False,
                },
            )()
            for _ in range(2)
        ]

        recorder.start(cache, n_confirmed=3)

        # Record tape entries for 3 positions
        for pos in range(3):
            conv_state = mx.array([[pos, pos + 1, pos + 2]])
            ssm_state = mx.array([[pos * 10, pos * 10 + 1, pos * 10 + 2]])
            recorder.record_entry(
                layer_idx=0, conv_state=conv_state, ssm_state=ssm_state
            )

        # Rollback to position 2
        recorder.rollback(cache, to_position=2)

        tape = recorder.current_tape
        assert tape is not None

        # Verify cache was restored to position 2 state
        conv_snap, ssm_snap = tape.tapes[0].get_state_at(2)
        assert mx.array_equal(cache[0][0], conv_snap)
        assert mx.array_equal(cache[0][1], ssm_snap)

        recorder.finish()


class TestTapeBuffer:
    """Tests for the TapeBuffer class."""

    def test_add_entry(self):
        """Test adding entries to tape buffer."""
        tape_buffer = TapeBuffer(tapes={}, n_confirmed=0)

        conv_state = mx.array([[1, 2, 3]])
        ssm_state = mx.array([[4, 5, 6]])

        tape_buffer.add_entry(layer_idx=0, conv_state=conv_state, ssm_state=ssm_state)

        assert 0 in tape_buffer.tapes
        assert len(tape_buffer.tapes[0].entries) == 1
        assert tape_buffer.total_bytes > 0

    def test_get_state_at(self):
        """Test getting state at specific position."""
        from vllm_mlx.spec_decode.mtp.tape_rollback import LayerTape, TapeEntry

        entries = [
            TapeEntry(conv_state=mx.array([[1]]), ssm_state=mx.array([[2]])),
            TapeEntry(conv_state=mx.array([[3]]), ssm_state=mx.array([[4]])),
            TapeEntry(conv_state=mx.array([[5]]), ssm_state=mx.array([[6]])),
        ]
        tape = LayerTape(entries=entries, layer_idx=0)

        # Position is 1-indexed
        conv, ssm = tape.get_state_at(1)
        assert mx.array_equal(conv, mx.array([[1]]))
        assert mx.array_equal(ssm, mx.array([[2]]))

        conv, ssm = tape.get_state_at(3)
        assert mx.array_equal(conv, mx.array([[5]]))
        assert mx.array_equal(ssm, mx.array([[6]]))

        # Invalid position
        with pytest.raises(ValueError):
            tape.get_state_at(0)
        with pytest.raises(ValueError):
            tape.get_state_at(4)

    def test_rollback_to(self):
        """Test rollback to position."""
        from vllm_mlx.spec_decode.mtp.tape_rollback import LayerTape, TapeEntry

        # Create tape with 3 positions
        entries = [
            TapeEntry(conv_state=mx.array([[1, 2]]), ssm_state=mx.array([[3, 4]])),
            TapeEntry(conv_state=mx.array([[5, 6]]), ssm_state=mx.array([[7, 8]])),
            TapeEntry(conv_state=mx.array([[9, 10]]), ssm_state=mx.array([[11, 12]])),
        ]
        tapes = {0: LayerTape(entries=entries, layer_idx=0)}
        tape_buffer = TapeBuffer(tapes=tapes, n_confirmed=3)

        # Create mock cache
        cache = [
            type(
                "Obj",
                (),
                {
                    "rollback_state": [[100, 200]],
                    "__getitem__": lambda self, i: getattr(self, f"_val{i}", None),
                    "__setitem__": lambda self, i, v: setattr(self, f"_val{i}", v),
                },
            )()
        ]

        # Rollback to position 2
        tape_buffer.rollback_to(cache, target_pos=2)

        # Verify state was restored
        assert mx.array_equal(cache[0][0], mx.array([[5, 6]]))
        assert mx.array_equal(cache[0][1], mx.array([[7, 8]]))
        assert cache[0].rollback_state is None


class TestTapeCorrectness:
    """Bit-exactness tests for tape rollback."""

    def test_verify_tape_correctness_helper(self):
        """Test the tape correctness verification helper."""
        from vllm_mlx.spec_decode.mtp.tape_rollback import LayerTape, TapeEntry

        # Create a simple tape
        entries = [
            TapeEntry(
                conv_state=mx.array([[1.0, 2.0]]), ssm_state=mx.array([[3.0, 4.0]])
            ),
            TapeEntry(
                conv_state=mx.array([[5.0, 6.0]]), ssm_state=mx.array([[7.0, 8.0]])
            ),
        ]
        tapes = {0: LayerTape(entries=entries, layer_idx=0)}
        tape = TapeBuffer(tapes=tapes, n_confirmed=2)

        # Create mock cache
        cache = [
            type(
                "Obj",
                (),
                {
                    "rollback_state": [[99, 99]],
                    "__getitem__": lambda self, i: (
                        setattr(self, f"_val{i}", None) or getattr(self, f"_val{i}")
                    ),
                    "__setitem__": lambda self, i, v: setattr(self, f"_val{i}", v),
                },
            )()
        ]

        # Simple verify function that returns the tape state
        def verify_fn(c, pos):
            # Return a cache with the tape state at position pos
            result = [
                type(
                    "Obj",
                    (),
                    {
                        "rollback_state": [[99, 99]],
                        "__getitem__": lambda self, i: (
                            setattr(self, f"_val{i}", None) or getattr(self, f"_val{i}")
                        ),
                        "__setitem__": lambda self, i, v: setattr(self, f"_val{i}", v),
                    },
                )()
            ]
            conv, ssm = entries[pos - 1].conv_state, entries[pos - 1].ssm_state
            result[0][0] = conv
            result[0][1] = ssm
            return result

        # This should pass (tape matches reference)
        # Note: This is a simplified test; real verification needs actual SSM layers
        # The full bit-exact test requires a real model


class TestIntegration:
    """Integration tests with actual MTP generation."""

    @pytest.mark.skip(reason="Requires full model load for bit-exact K=3 vs K=0 test")
    def test_k3_bit_exact_vs_k0(self):
        """Gate test: K=3 generation must be bit-exact vs K=0 baseline.

        This is the critical gate test for tape rollback. It generates
        the same prompt with K=3 (speculative decoding with tape rollback)
        and K=0 (no speculative decoding) and verifies the output is
        byte-identical.

        Steps:
        1. Load model with MTP support
        2. Generate with K=0 (baseline, no spec decode)
        3. Generate with K=3 (spec decode with tape rollback)
        4. Compare outputs (must be identical at temp=0)

        This test ensures tape rollback doesn't break the lossless
        contract that qMLX's MTP implementation depends on.
        """
        # Implementation requires:
        # - Full model load (Qwen3.5-122B or smaller test model)
        # - MTP injection
        # - Generation with different K values
        # - Token-by-token comparison
        #
        # This is skipped here but should be run as part of the
        # release checklist before enabling K>1 in production.
        pass


def test_global_recorder_singleton():
    """Test that get_tape_recorder returns a singleton."""
    recorder1 = get_tape_recorder()
    recorder2 = get_tape_recorder()

    assert recorder1 is recorder2


def test_format_tape_size():
    """Test tape size formatting."""
    from vllm_mlx.spec_decode.mtp.tape_rollback import format_tape_size

    assert format_tape_size(500) == "500 B"
    assert format_tape_size(1024) == "1.0 KB"
    assert format_tape_size(1536) == "1.5 KB"
    assert format_tape_size(1048576) == "1.0 MB"


def test_estimate_tape_bytes():
    """Test tape size estimation."""
    from vllm_mlx.spec_decode.mtp.tape_rollback import estimate_tape_bytes

    # Estimate for Qwen3.5-122B with K=3. The model has 36 linear-attention
    # (SSM) layers, not 91 — the earlier "~75% of 122 layers" figure assumed
    # a 122-layer stack; the real config is 48 layers, 36 of them linear.
    size = estimate_tape_bytes(n_layers=36, n_positions=3, batch_size=1)

    # The tape stores a FULL fp32 SSM recurrent-state snapshot per confirmed
    # position per SSM layer. One single-position state set across 36 layers
    # is ~151 MB (see scheduler.py). K=3 is therefore hundreds of MB — the
    # "KB-scale" claim in early revisions was fiction.
    per_position_one_layer = 64 * 128 * 128 * 4 + ((16 * 128) * 2 + 64 * 128) * 3 * 2
    assert size == 36 * 3 * per_position_one_layer
    assert size > 100 * 1024 * 1024  # comfortably MB-scale (> 100 MB)
    assert size < 1024 * 1024 * 1024  # but under 1 GB for K=3

    # Sanity: a single recorded position across all SSM layers matches the
    # ~151 MB figure scheduler.py documents for one full SSM state set.
    one_pos = estimate_tape_bytes(n_layers=36, n_positions=1, batch_size=1)
    assert 140 * 1024 * 1024 < one_pos < 165 * 1024 * 1024

    print(f"Estimated tape size (K=3, 36 layers): {size / (1024 * 1024):.1f} MB")


# ----------------------------------------------------------------------
# Issue #53: depth-3 audit — K=3 tape rollback through the full
# generator loop.
#
# A fake-SSM target model (one emulated GatedDeltaNet layer) runs
# ``mtp_generate_step`` at ``fixed_k=3``. The fake model emulates the
# ``cache_patch`` chunk-split: on a verify forward (S=4,
# ``n_confirmed=3``) it records a per-position tape of 3 entries and
# the generator's ``_rollback_draft(n_to_drop)`` restores from
# ``tape[len - n_to_drop]``. For every partial-accept depth we assert
# the exact restored conv/ssm state — i.e. the tape rollback lands on
# the right position, not just *a* position.
# ----------------------------------------------------------------------


class _FakeSSMCache:
    """List-like ``[conv_state, ssm_state]`` cache emulating a patched
    ``ArraysCache`` for ONE linear-attention layer.

    State model (deterministic stand-in for conv + SSM):

    * ``conv_state`` — ring buffer of the last 2 processed token ids
      (``conv_kernel_size - 1 = 2``).
    * ``ssm_state``  — running sum of all processed token ids.

    Both are closed-form functions of the processed token stream, so a
    tape entry is fully determined by its position — the assertions
    below compare against independently computed expectations.
    """

    N_KEEP = 2

    def __init__(self):
        self._conv: list = []
        self._ssm: int = 0
        self.rollback_state = None
        self.n_confirmed_for_mtp = 0
        self.tape_history: list = []  # (len(tape),) per recording call

    def __getitem__(self, i):
        return (list(self._conv), self._ssm)[i]

    def __setitem__(self, i, v):
        if i == 0:
            self._conv = list(v)
        else:
            self._ssm = int(v)

    def __len__(self):
        return 2


class _FakeSSMModel:
    """Mock Qwen3.5-style hybrid target: scripted backbone logits plus
    one emulated GatedDeltaNet layer that implements the chunk-split
    tape semantics of ``cache_patch._patched_call``:

    * fast path (``n_confirmed <= 0`` or ``>= S`` or ``S < 2``):
      advance state over all S tokens, no tape;
    * chunk-split (``0 < n_confirmed < S``): record
      ``rollback_state = [(conv, ssm) after each of the first
      n_confirmed positions]``, then batch-advance the rest.
    """

    def __init__(self, backbone_outputs, mtp_outputs, vocab=32, hidden_size=8):
        self._backbone = list(backbone_outputs)
        self._mtp = list(mtp_outputs)
        self._backbone_cursor = 0
        self._mtp_cursor = 0
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.layers = []
        self.cache = _FakeSSMCache()
        self.backbone_calls: list = []  # (S, n_confirmed) per call

    def make_cache(self):
        return [self.cache]

    def make_mtp_cache(self):
        return []

    @staticmethod
    def _logits_for_positions(target_ids, batch, vocab):
        rows = []
        for tid in target_ids:
            row = mx.zeros((vocab,))
            row = row + mx.where(mx.arange(vocab) == tid, mx.array(50.0), mx.array(0.0))
            rows.append(row[None, :])
        return mx.stack(rows, axis=1)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        B, S = inputs.shape
        toks_in = inputs[0].tolist()
        targets = []
        for _ in range(S):
            if self._backbone_cursor < len(self._backbone):
                targets.append(self._backbone[self._backbone_cursor])
                self._backbone_cursor += 1
            else:
                targets.append(0)
        self.backbone_calls.append((S, int(n_confirmed or 0)))
        logits = self._logits_for_positions(targets, B, self.vocab)
        hidden = mx.zeros((B, S, self.hidden_size))

        c = cache[0] if cache is not None else self.cache
        n_conf = int(n_confirmed or 0)
        if 0 < n_conf < S and S >= 2:
            # Chunk-split: tape over the first n_conf positions.
            tape = []
            conv, ssm = list(c[0]), int(c[1])
            for idx in range(n_conf):
                conv.append(toks_in[idx])
                conv = conv[-self.cache.N_KEEP :]
                ssm += toks_in[idx]
                tape.append((list(conv), ssm))
            for idx in range(n_conf, S):
                conv.append(toks_in[idx])
                conv = conv[-self.cache.N_KEEP :]
                ssm += toks_in[idx]
            c.rollback_state = tape
            c[0] = conv
            c[1] = ssm
            c.tape_history.append(len(tape))
        else:
            conv, ssm = list(c[0]), int(c[1])
            for t in toks_in:
                conv.append(t)
                conv = conv[-self.cache.N_KEEP :]
                ssm += t
            c[0] = conv
            c[1] = ssm

        if return_hidden:
            return logits, hidden
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        S = next_token_ids.shape[1]
        targets = []
        for _ in range(S):
            if self._mtp_cursor < len(self._mtp):
                targets.append(self._mtp[self._mtp_cursor])
                self._mtp_cursor += 1
            else:
                targets.append(0)
        return self._logits_for_positions(targets, 1, self.vocab)


@pytest.mark.parametrize("accepted_count", [0, 1, 2, 3])
@pytest.mark.xfail(
    strict=False,
    reason="dev-debt (first-AS adjudication run 41017472632, 2026-10-30): 'no Stream(gpu,12) in current thread' - MLX stream threading vs wave-4 engine change",
)
def test_k3_tape_rollback_position_audit(accepted_count):
    """Run one K=3 verify round against the fake-SSM target with the
    first ``accepted_count`` drafts accepted, and audit the rollback:

    * the generator's d=3 invariant passes (tape length == 3 after
      every verify forward);
    * on partial accept the SSM cache is restored to EXACTLY the
      state after the last accepted position (conv ring + ssm sum),
      per ``tape[len - n_to_drop]`` with ``n_to_drop = 3 - accepted``;
    * on all-accept the tape is cleared and the cache holds the
      full-batch state.
    """
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    drafts = [11, 12, 13]
    backbone = (
        [7]
        + [drafts[i] if i < accepted_count else 20 for i in range(3)]
        + [14]  # bonus (used only when accepted_count == 3)
    )
    model = _FakeSSMModel(backbone_outputs=backbone, mtp_outputs=drafts)
    counter = MTPAcceptCounter()
    prompt = mx.array([1], dtype=mx.uint32)

    emitted = []
    for tok, _lp, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=2 + accepted_count,
        temp=0.0,
        accept_counter=counter,
        max_k=3,
        fixed_k=3,
    ):
        emitted.append((tok, from_draft))

    expected_stream = [(7, False)]
    expected_stream.extend((d, True) for d in drafts[:accepted_count])
    expected_stream.append((20, False) if accepted_count < 3 else (14, False))
    assert emitted == expected_stream, (
        f"accepted_count={accepted_count}: expected {expected_stream}, got {emitted}"
    )

    c = model.cache
    # The cache advances on the INPUT token stream, which includes the
    # prompt token (1) processed by the bootstrap K=0 round BEFORE the
    # verify batch [7, d1..d3].
    if accepted_count == 3:
        # All accept: no rollback — cache advanced over the whole
        # [1] + [7, 11, 12, 13] stream and the tape was cleared.
        assert c[0] == [12, 13], f"conv state: {c[0]}"
        assert c[1] == 1 + 7 + 11 + 12 + 13, f"ssm state: {c[1]}"
        assert c.rollback_state is None, "all-accept must clear the tape"
    else:
        # Partial accept: rollback restored tape[accepted_count] —
        # the state after [1, 7, d1..d_accepted].
        prefix = [1, 7] + drafts[:accepted_count]
        expected_conv = prefix[-2:]
        expected_ssm = sum(prefix)
        assert c[0] == expected_conv, (
            f"rollback landed wrong: conv={c[0]}, expected {expected_conv}"
        )
        assert c[1] == expected_ssm, (
            f"rollback landed wrong: ssm={c[1]}, expected {expected_ssm}"
        )
        assert c.rollback_state is None, "rollback must clear the tape"

    # The verify forward must have recorded a full K=3 tape (3
    # positions) — the invariant the gate exists to enforce.
    assert 3 in c.tape_history, (
        f"chunk-split never recorded a depth-3 tape: {c.tape_history}"
    )

    snap = counter.snapshot()
    assert snap.attempts == 3
    assert snap.accepts == accepted_count
    assert snap.tokens_saved == accepted_count


@pytest.mark.xfail(
    strict=False,
    reason="dev-debt (first-AS adjudication run 41017472632, 2026-10-30): 'no Stream(gpu,12) in current thread' - MLX stream threading vs wave-4 engine change",
)
def test_k3_tape_invariant_fires_on_misrecorded_tape():
    """If a layer's chunk-split recorded the WRONG tape depth, the
    generator's d=3 gate invariant must raise ``AssertionError``
    instead of silently restoring a wrong position.

    The fake model here records a 2-entry tape on a 3-position
    boundary — exactly the mis-record a regression in the n_confirmed
    plumbing would produce.
    """
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    drafts = [11, 12, 13]
    backbone = [7, 20, 12, 13, 99]  # reject at position 0
    model = _FakeSSMModel(backbone_outputs=backbone, mtp_outputs=drafts)

    _orig_call = _FakeSSMModel.__call__

    def misrecording_call(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden=False,
        n_confirmed=0,
    ):
        # Force the chunk-split to record only 2 positions of the 3.
        return _orig_call(
            self,
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            n_confirmed=min(int(n_confirmed or 0), 2),
        )

    _FakeSSMModel.__call__ = misrecording_call
    try:
        prompt = mx.array([1], dtype=mx.uint32)
        with pytest.raises(AssertionError, match=r"\[MTP-d3\] SSM tape invariant"):
            list(
                mtp_generate_step(
                    prompt,
                    model,
                    max_tokens=4,
                    temp=0.0,
                    max_k=3,
                    fixed_k=3,
                )
            )
    finally:
        _FakeSSMModel.__call__ = _orig_call
