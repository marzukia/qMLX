# SPDX-License-Identifier: Apache-2.0
"""
Incremental token decoder with multi-byte character safety.

This module provides ``IncrementalDecoder``, used by the Scheduler to safely decode
streaming tokens without emitting broken multi-byte characters (emoji, CJK).

t3219 F1a: ``add_token`` is now TAIL-ONLY — it re-decodes only the last
``_WIN_CAP`` token ids per step (O(1)) instead of the entire accumulated
history (O(T) per step → O(T^2) per request, the named 2.4x serve decode
gap). The U+FFFD hold-back contract is preserved. Requires a tokenizer
whose ``decode`` is a byte/text concatenation of per-token decodes
(byte-level BPE fast tokenizers with clean-up off, e.g. Qwen3) — verified
for the target model (concatenation property + ``clean_up_tokenized_output``
off). Genuine mid-stream U+FFFD characters (invalid bytes, e.g. 0xFF) are
now emitted immediately instead of freezing the stream as the old
full-redecode ``"\ufffd" in delta`` check did.
"""


class IncrementalDecoder:
    """Decode tokens incrementally with multi-byte character safety.

    t3219 F1a: instead of re-decoding the whole accumulated token list on
    every step, only a bounded trailing window (``_WIN_CAP`` ids) is
    re-decoded; the emitted delta is a plain string slice against the
    previous window's decode (a strict prefix window, so the prefix
    property holds). If the window's decode ends in U+FFFD (incomplete
    multi-byte sequence), output is held back until the character
    completes.

    Example::

        decoder = IncrementalDecoder(tokenizer, skip_special_tokens=False)
        for tok_id in stream:
            delta = decoder.add_token(tok_id)
            if delta:
                send_to_client(delta)
    """

    # UTF-8 sequences span at most 4 bytes; 8 ids is a generous cap.
    _WIN_CAP = 8

    __slots__ = (
        "_tokenizer",
        "_skip_special_tokens",
        "_token_ids",
        "_win",
        "_win_text",
    )

    def __init__(self, tokenizer, skip_special_tokens: bool = False):
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._token_ids: list[int] = []
        # t3219 F1a tail-only window state.
        self._win: list[int] = []
        self._win_text: str = ""

    def add_token(self, token_id: int) -> str:
        """Add a token and return the safe new text delta.

        Returns an empty string if the window's decode ends in U+FFFD
        (incomplete trailing multi-byte sequence), deferring output until
        the character completes.

        Invariant: ``_win_text`` is always the decode of a strict prefix
        of ``_win`` from a previous step where the decode had no trailing
        U+FFFD, so ``_win_text`` is a prefix of the current window decode
        and the delta is an exact string slice — no full-history
        re-decode on the hot path.
        """
        self._token_ids.append(token_id)
        self._win.append(token_id)
        t = self._tokenizer.decode(
            self._win, skip_special_tokens=self._skip_special_tokens
        )
        if t.endswith("\ufffd"):
            # Incomplete trailing multi-byte sequence: hold back until more
            # tokens complete the character. (Do NOT shrink/re-anchor: the
            # anchor must stay a no-FFFD decode.)
            return ""
        delta = t[len(self._win_text) :]
        if len(self._win) > self._WIN_CAP:
            # The leftmost id's bytes are complete (no trailing U+FFFD):
            # confirm it and re-anchor on the smaller window (O(1): decodes
            # <= _WIN_CAP ids).
            self._win = self._win[1:]
            self._win_text = self._tokenizer.decode(
                self._win, skip_special_tokens=self._skip_special_tokens
            )
        else:
            self._win_text = t
        return delta

    def get_full_text(self) -> str:
        """Get the full decoded text so far (re-decodes all tokens).

        One-shot O(T) — used only at request finish (final
        ``output_text`` flush) and stop-string matching, never on the
        per-step path.
        """
        if not self._token_ids:
            return ""
        return self._tokenizer.decode(
            self._token_ids, skip_special_tokens=self._skip_special_tokens
        )

    @property
    def token_ids(self) -> list[int]:
        """Read-only access to accumulated token IDs."""
        return self._token_ids

    @property
    def prev_text(self) -> str:
        """Streaming surface for the stop-string trim path (t3219 F1a).

        Only consumed when user-supplied stop params are active AND a
        hold-back step is in flight — never on the per-step hot path.
        Implemented as a one-shot full re-decode (O(T)); equals
        ``get_full_text()`` up to the held-back trailing bytes.
        """
        return self.get_full_text()

    def reset(self):
        """Reset state for a new sequence."""
        self._token_ids.clear()
        self._win.clear()
        self._win_text = ""
