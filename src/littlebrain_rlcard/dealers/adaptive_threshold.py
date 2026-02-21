"""
AdaptiveThresholdDealer — Algorithm 3.1 from arXiv:2505.01287.

Two-phase algorithm:
  1. Adaptive-threshold phase (t = 1 .. n - 2d): sample from mini-decks with
     rejection based on an adaptive threshold.
  2. Final phase: uniform without replacement from the ≤ 2d remaining cards
     via swap-delete.

Parameters
----------
m_bits : int
    Memory budget in bits;  ``d = max(1, floor(m_bits / 8))`` by default.
encoding : ``"naive"`` | ``"holes_elias_doc"``
    * ``"naive"`` — store ``ell[i]`` counters explicitly (easiest).
    * ``"holes_elias_doc"`` — same runtime, but docstrings explain how the
      paper would encode hole distances ``x_i = threshold - ell[i]`` with
      Elias-gamma codes to achieve O(m) total bits.
"""

from __future__ import annotations

import math

from littlebrain_rlcard.dealers.common import BaseDealer, _deep_getsizeof, uniform_int


def _elias_gamma_bits(x: int) -> int:
    """Estimated bit-length of Elias-gamma encoding of *x* (x ≥ 1).

    Used only for the ``"holes_elias_doc"`` accounting; the runtime behaviour
    is identical to ``"naive"``.

    Elias-gamma encodes positive integer *x* in ``2*floor(log2(x)) + 1`` bits.
    Convention: we encode ``x_i + 1`` (since x_i can be 0).
    """
    val = x + 1  # shift so minimum is 1
    return 2 * math.floor(math.log2(val)) + 1 if val >= 1 else 1


class AdaptiveThresholdDealer(BaseDealer):
    """Adaptive-threshold dealer (Algorithm 3.1).

    After :meth:`reset`, call :meth:`draw` exactly *n* times to obtain a
    permutation of ``{0, …, n-1}``.
    """

    def __init__(self) -> None:
        # Per-reset state
        self._d: int = 0
        self._sizes: list[int] = []
        self._starts: list[int] = []
        self._ell: list[int] = []
        self._t: int = 0
        self._phase: str = "adaptive"  # or "final"
        self._final_cards: list[int] = []
        self._final_remaining: int = 0
        self._encoding: str = "naive"
        self._m_bits: int = 64

    # -- BaseDealer interface ------------------------------------------------

    def reset(self, n: int, np_random, **params) -> None:
        self._n = n
        self._np_random = np_random
        self._num_drawn = 0

        self._m_bits = params.get("m_bits", 64)
        self._encoding = params.get("encoding", "naive")
        d = max(1, self._m_bits // 8)
        if d > n // 2:
            d = max(1, n // 2)
        self._d = d

        # Partition n into d mini-decks with sizes differing by at most 1
        base_size = n // d
        extra = n % d
        self._sizes = []
        self._starts = []
        offset = 0
        for i in range(d):
            sz = base_size + (1 if i < extra else 0)
            self._sizes.append(sz)
            self._starts.append(offset)
            offset += sz

        self._ell = [0] * d
        self._t = 0
        self._phase = "adaptive"
        self._final_cards = []
        self._final_remaining = 0

    def draw(self) -> int:
        self._check_exhausted()
        if self._phase == "adaptive":
            return self._draw_adaptive()
        else:
            return self._draw_final()

    def remaining(self) -> int:
        return self._n - self._num_drawn

    def state_summary(self) -> dict:
        if self._encoding == "holes_elias_doc":
            threshold = self._current_threshold()
            total_elias = sum(
                _elias_gamma_bits(threshold - self._ell[i])
                for i in range(self._d)
                if self._ell[i] < self._sizes[i]
            )
            theory_bits = total_elias + self._d * 2  # + overhead
        else:
            bits_per_ell = max(1, math.ceil(math.log2(max(self._n, 2))))
            theory_bits = self._d * bits_per_ell
        return {
            "algorithm": "AdaptiveThresholdDealer",
            "n": self._n,
            "d": self._d,
            "m_bits": self._m_bits,
            "encoding": self._encoding,
            "drawn": self._num_drawn,
            "remaining": self.remaining(),
            "phase": self._phase,
            "t": self._t,
            "theoretical_bits": theory_bits,
            "python_bytes": (
                _deep_getsizeof(self._ell)
                + _deep_getsizeof(self._sizes)
                + _deep_getsizeof(self._starts)
                + _deep_getsizeof(self._final_cards)
            ),
        }

    def peek_next_distribution(self) -> dict[int, float] | None:
        if self.remaining() == 0:
            return None
        if self._phase == "final":
            # uniform over final_cards
            rem = self._final_remaining
            if rem == 0:
                return None
            prob = 1.0 / rem
            return {self._final_cards[i]: prob for i in range(rem)}
        # Adaptive phase: each drawable mini-deck contributes its top card
        drawable = self._get_drawable_indices()
        if not drawable:
            return None
        prob = 1.0 / len(drawable)
        return {self._top_card(i): prob for i in drawable}

    def peek_drawable_options(self) -> list[tuple[int, float]]:
        """Return ``[(top_card_id, probability), ...]`` for the current step.

        Useful for scripts and tests that inspect the adaptive phase.
        """
        dist = self.peek_next_distribution()
        if dist is None:
            return []
        return [(cid, p) for cid, p in dist.items()]

    # -- internals -----------------------------------------------------------

    def _current_threshold(self) -> int:
        if self._t == 0:
            return 1
        return math.ceil(self._t / self._d) + 1

    def _top_card(self, i: int) -> int:
        """Card id at the top of mini-deck *i*."""
        return self._starts[i] + self._ell[i]

    def _get_drawable_indices(self) -> list[int]:
        """Return list of mini-deck indices that are drawable at current t."""
        threshold = self._current_threshold()
        return [
            i for i in range(self._d)
            if self._ell[i] < threshold and self._ell[i] < self._sizes[i]
        ]

    def _draw_adaptive(self) -> int:
        self._t += 1
        n_adaptive = self._n - 2 * self._d
        if self._t > n_adaptive:
            # Transition to final phase
            self._transition_to_final()
            return self._draw_final()

        threshold = self._current_threshold()
        # Rejection sampling over mini-decks
        while True:
            i = uniform_int(self._np_random, 0, self._d - 1)
            if self._ell[i] < threshold and self._ell[i] < self._sizes[i]:
                card = self._top_card(i)
                self._ell[i] += 1
                self._num_drawn += 1
                return card

    def _transition_to_final(self) -> None:
        """Build the remaining-cards list and switch to final phase."""
        self._phase = "final"
        self._final_cards = []
        for i in range(self._d):
            start = self._starts[i] + self._ell[i]
            end = self._starts[i] + self._sizes[i]
            for cid in range(start, end):
                self._final_cards.append(cid)
        self._final_remaining = len(self._final_cards)

    def _draw_final(self) -> int:
        """Swap-delete from the final-cards array."""
        if self._final_remaining == 0:
            raise RuntimeError("Final phase exhausted unexpectedly")
        i = uniform_int(self._np_random, 0, self._final_remaining - 1)
        out = self._final_cards[i]
        self._final_cards[i] = self._final_cards[self._final_remaining - 1]
        self._final_remaining -= 1
        self._num_drawn += 1
        return out
