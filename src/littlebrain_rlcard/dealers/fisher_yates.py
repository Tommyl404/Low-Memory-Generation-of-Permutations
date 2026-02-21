"""
FisherYatesDealer — Algorithm 2.2 from arXiv:2505.01287 (Knuth's dealer).

Classic swap-delete (inside-out Fisher–Yates) that generates a uniform random
permutation one element at a time.  O(1) per draw, O(n) total memory.
"""

from __future__ import annotations

import math

from littlebrain_rlcard.dealers.common import BaseDealer, _deep_getsizeof, uniform_int


class FisherYatesDealer(BaseDealer):
    """Swap-delete dealer (Fisher–Yates / Knuth shuffle)."""

    def __init__(self) -> None:
        self._array: list[int] = []
        self._remaining: int = 0

    # -- BaseDealer interface ------------------------------------------------

    def reset(self, n: int, np_random, **params) -> None:
        self._n = n
        self._np_random = np_random
        self._num_drawn = 0
        self._array = list(range(n))
        self._remaining = n

    def draw(self) -> int:
        self._check_exhausted()
        i = uniform_int(self._np_random, 0, self._remaining - 1)
        out = self._array[i]
        self._array[i] = self._array[self._remaining - 1]
        self._remaining -= 1
        self._num_drawn += 1
        return out

    def remaining(self) -> int:
        return self._remaining

    def state_summary(self) -> dict:
        n_bits = self._n * max(1, math.ceil(math.log2(max(self._n, 2))))
        return {
            "algorithm": "FisherYatesDealer",
            "n": self._n,
            "drawn": self._num_drawn,
            "remaining": self._remaining,
            "theoretical_bits": n_bits,
            "python_bytes": _deep_getsizeof(self._array),
        }

    def peek_next_distribution(self) -> dict[int, float] | None:
        rem = self.remaining()
        if rem == 0:
            return None
        prob = 1.0 / rem
        return {self._array[i]: prob for i in range(rem)}
