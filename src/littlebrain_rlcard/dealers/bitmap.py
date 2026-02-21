"""
BitmapDealer — Algorithm 2.1 from arXiv:2505.01287.

Rejection-sampling against an availability bitmap.  Uniform but can be slow
for large *n* near the end of the deck (expected draws per card grow as the
deck thins).  Fine for n ≤ 104.
"""

from __future__ import annotations

from littlebrain_rlcard.dealers.common import BaseDealer, _deep_getsizeof, uniform_int


class BitmapDealer(BaseDealer):
    """Rejection-sampling dealer using a boolean availability bitmap."""

    def __init__(self) -> None:
        self._available: bytearray = bytearray()

    # -- BaseDealer interface ------------------------------------------------

    def reset(self, n: int, np_random, **params) -> None:
        self._n = n
        self._np_random = np_random
        self._num_drawn = 0
        self._available = bytearray(b"\x01" * n)

    def draw(self) -> int:
        self._check_exhausted()
        while True:
            c = uniform_int(self._np_random, 0, self._n - 1)
            if self._available[c]:
                self._available[c] = 0
                self._num_drawn += 1
                return c

    def remaining(self) -> int:
        return self._n - self._num_drawn

    def state_summary(self) -> dict:
        return {
            "algorithm": "BitmapDealer",
            "n": self._n,
            "drawn": self._num_drawn,
            "remaining": self.remaining(),
            "theoretical_bits": self._n,  # 1 bit per card
            "python_bytes": _deep_getsizeof(self._available),
        }

    def peek_next_distribution(self) -> dict[int, float] | None:
        rem = self.remaining()
        if rem == 0:
            return None
        prob = 1.0 / rem
        return {i: prob for i in range(self._n) if self._available[i]}
