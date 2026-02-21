"""
Base class and helpers shared by all dealer implementations.

References
----------
Menuhin & Naor, "How to Shuffle in Sublinear Memory", arXiv:2505.01287.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uniform_int(np_random, low_inclusive: int, high_inclusive: int) -> int:
    """Return a uniformly random integer in [low_inclusive, high_inclusive].

    Uses *only* the provided ``np_random`` generator (never global state).
    Wraps ``np_random.randint(low, high_exclusive)``.
    """
    return int(np_random.randint(low_inclusive, high_inclusive + 1))


def popcount(x: int) -> int:
    """Population count (number of set bits) — delegates to ``int.bit_count()``."""
    return x.bit_count()


def bit_select(mask: int, r: int) -> int:
    """Return the index of the *r*-th set bit (0-based) in *mask*.

    Implements a simple loop over set bits.  This is fast enough for cell
    widths w ≤ 8 (PerfectDealer with n ≤ 256).  For large *n*, a broadword /
    PDEP-based select would be needed — see Vigna "broadword select".
    """
    count = 0
    pos = 0
    while mask:
        if mask & 1:
            if count == r:
                return pos
            count += 1
        mask >>= 1
        pos += 1
    raise ValueError(f"bit_select: mask has fewer than {r + 1} set bits")


def _deep_getsizeof(obj: object, seen: set | None = None) -> int:
    """Recursively estimate memory usage of *obj* in bytes."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_getsizeof(k, seen) + _deep_getsizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_getsizeof(item, seen)
    elif isinstance(obj, bytearray):
        pass  # sys.getsizeof already includes buffer
    return size


# ---------------------------------------------------------------------------
# Base dealer
# ---------------------------------------------------------------------------

class BaseDealer(ABC):
    """Abstract base class for all dealer algorithms.

    Subclasses must implement :meth:`reset`, :meth:`draw`, :meth:`remaining`,
    and :meth:`state_summary`.
    """

    _n: int = 0
    _np_random: Any = None
    _num_drawn: int = 0

    # -- required interface --------------------------------------------------

    @abstractmethod
    def reset(self, n: int, np_random, **params) -> None:  # noqa: ANN001
        """Initialise (or re-initialise) the dealer for a fresh permutation.

        Parameters
        ----------
        n : int
            Number of elements (cards) to shuffle.
        np_random : numpy.random.RandomState
            The *only* source of randomness the dealer may use.
        **params :
            Algorithm-specific configuration.
        """

    @abstractmethod
    def draw(self) -> int:
        """Draw (deal) the next card id ∈ [0, n-1].

        Must raise ``RuntimeError`` if deck is exhausted.
        """

    def next_card(self) -> int:
        """Alias for :meth:`draw`."""
        return self.draw()

    @abstractmethod
    def remaining(self) -> int:
        """Number of cards still available to draw."""

    @abstractmethod
    def state_summary(self) -> dict:
        """Return a dict summarising the dealer's internal state.

        Must include at least ``'theoretical_bits'`` (int) and
        ``'python_bytes'`` (int).
        """

    # -- optional interface --------------------------------------------------

    def peek_next_distribution(self) -> dict[int, float] | None:
        """Return the probability distribution over possible next draws.

        Returns ``None`` if not implemented / not meaningful.  For uniform
        dealers this is ``{id: 1/remaining}`` over remaining card ids.
        """
        return None

    # -- helpers -------------------------------------------------------------

    def _check_exhausted(self) -> None:
        if self._num_drawn >= self._n:
            raise RuntimeError(
                f"Deck exhausted: already drew {self._num_drawn}/{self._n} cards"
            )
