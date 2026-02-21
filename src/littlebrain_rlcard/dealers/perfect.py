"""
PerfectDealer — Appendix A + Section 4.1 from arXiv:2505.01287.

Cells / Intervals / Population structure for *n* ≤ 256 (w ≤ 8).

Overview
--------
Elements ``{0, …, n-1}`` are partitioned into *C = ⌈n/w⌉* cells, each a
*w*-bit mask.  Cells are arranged in contiguous intervals by population
(number of set bits).  Drawing a card involves:

1. **Population sampling** — choose population *p* with probability
   proportional to ``p × interval_size[p]``.  Implemented here as a
   prefix-sum scan (O(w)).  For large *n*, Section 4.2 describes a
   constant-time dynamic pseudo-distribution sampler (Urn + residue tables).
2. **Cell sampling** — pick a random cell within the chosen interval.
3. **Element sampling** — pick a random set bit inside the cell via
   ``bit_select``.
4. **Interval update** — ``DecrementCellPopulationSize`` (Alg. A.2): swap the
   cell to the boundary of its interval and shrink/grow the adjacent interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from littlebrain_rlcard.dealers.common import (
    BaseDealer,
    _deep_getsizeof,
    bit_select,
    popcount,
    uniform_int,
)


@dataclass
class _Cell:
    """A single cell in the PerfectDealer structure."""
    index: int       # cell index j
    mask: int        # w-bit availability mask
    base: int        # first element id = j * w

    @property
    def pop(self) -> int:
        return popcount(self.mask)


class PerfectDealer(BaseDealer):
    """Optimal-entropy dealer using cells and population intervals.

    Produces a uniformly random permutation of ``{0, …, n-1}`` using O(n) bits
    of state and O(1) *amortised* random bits per draw (for large *n*).

    .. note::
       The PopSampler used here is a simple O(w) prefix-sum scan, suitable for
       *n* ≤ 256 (w ≤ 8).  For larger *n*, Section 4.2 of the paper describes
       a constant-time dynamic sampler.
    """

    def __init__(self) -> None:
        self._w: int = 1
        self._num_cells: int = 0
        self._cells: list[_Cell] = []
        # Interval bookkeeping: cells are stored in self._cells such that
        # cells with population p occupy indices
        #   [interval_begin[p] .. interval_begin[p] + interval_size[p])
        self._interval_begin: list[int] = []
        self._interval_size: list[int] = []

    # -- BaseDealer interface ------------------------------------------------

    def reset(self, n: int, np_random, **params) -> None:
        self._n = n
        self._np_random = np_random
        self._num_drawn = 0

        self._w = max(1, math.ceil(math.log2(max(n, 2))))
        num_cells = math.ceil(n / self._w)
        self._num_cells = num_cells
        w = self._w

        # Build cells
        self._cells = []
        for j in range(num_cells):
            base = j * w
            # Number of valid bits: min(w, n - base)
            valid = min(w, n - base)
            mask = (1 << valid) - 1  # all valid bits set
            self._cells.append(_Cell(index=j, mask=mask, base=base))

        # Sort cells by population (all start full, but last cell may differ)
        self._cells.sort(key=lambda c: c.pop)

        # Build interval arrays (population 0 .. w)
        self._interval_begin = [0] * (w + 1)
        self._interval_size = [0] * (w + 1)

        # Count populations
        pop_counts: dict[int, int] = {}
        for c in self._cells:
            p = c.pop
            pop_counts[p] = pop_counts.get(p, 0) + 1

        # Assign intervals (populations in increasing order, contiguously)
        offset = 0
        for p in range(w + 1):
            cnt = pop_counts.get(p, 0)
            self._interval_begin[p] = offset
            self._interval_size[p] = cnt
            offset += cnt

    def draw(self) -> int:
        self._check_exhausted()

        w = self._w

        # 1) Population sampling via prefix-sum scan  (O(w))
        # a[p] = p * interval_size[p]  for p in 1..w
        total = 0
        for p in range(1, w + 1):
            total += p * self._interval_size[p]

        if total == 0:
            raise RuntimeError("No drawable cells — should not happen")

        r = uniform_int(self._np_random, 0, total - 1)
        chosen_pop = 0
        cumul = 0
        for p in range(1, w + 1):
            cumul += p * self._interval_size[p]
            if r < cumul:
                chosen_pop = p
                break

        # 2) Cell sampling: random cell in the interval for chosen_pop
        isize = self._interval_size[chosen_pop]
        loc = uniform_int(self._np_random, 0, isize - 1)
        cell_idx = self._interval_begin[chosen_pop] + loc
        cell = self._cells[cell_idx]

        # 3) Element sampling: random set bit in cell mask
        bit_r = uniform_int(self._np_random, 0, chosen_pop - 1)
        bit_pos = bit_select(cell.mask, bit_r)
        element_id = cell.base + bit_pos

        # Clear the bit
        cell.mask &= ~(1 << bit_pos)

        # 4) Interval update: DecrementCellPopulationSize (Alg. A.2)
        self._decrement_cell_population(cell_idx, chosen_pop)

        self._num_drawn += 1
        return element_id

    def remaining(self) -> int:
        return self._n - self._num_drawn

    def state_summary(self) -> dict:
        # Each cell stores a w-bit mask; total = C * w bits ≈ n bits
        theory_bits = self._num_cells * self._w
        py_bytes = _deep_getsizeof(self._cells) + _deep_getsizeof(
            self._interval_begin
        ) + _deep_getsizeof(self._interval_size)
        return {
            "algorithm": "PerfectDealer",
            "n": self._n,
            "w": self._w,
            "num_cells": self._num_cells,
            "drawn": self._num_drawn,
            "remaining": self.remaining(),
            "theoretical_bits": theory_bits,
            "python_bytes": py_bytes,
        }

    def peek_next_distribution(self) -> dict[int, float] | None:
        rem = self.remaining()
        if rem == 0:
            return None
        # Uniform: each remaining element has prob 1/remaining
        prob = 1.0 / rem
        dist: dict[int, float] = {}
        for cell in self._cells:
            mask = cell.mask
            pos = 0
            while mask:
                if mask & 1:
                    dist[cell.base + pos] = prob
                mask >>= 1
                pos += 1
        return dist

    # -- internals -----------------------------------------------------------

    def _decrement_cell_population(self, cell_idx: int, old_pop: int) -> None:
        """Move cell from interval *old_pop* to interval *old_pop - 1*.

        Implements Algorithm A.2 (DecrementCellPopulationSize):
          - Swap cell at *cell_idx* with the *first* cell in interval *old_pop*.
          - Shrink interval *old_pop* from the left.
          - Grow interval *old_pop - 1* to the right.

        Intervals are laid out contiguously in increasing population order:
        ``[pop0 cells][pop1 cells]...[popW cells]``.  The cell that moves from
        pop *p* to pop *p-1* is placed at the boundary between the two
        intervals, preserving the contiguity invariant.
        """
        new_pop = old_pop - 1

        # First position in the old_pop interval
        first_in_interval = self._interval_begin[old_pop]

        # Swap the drawn cell with the first cell in its interval
        if cell_idx != first_in_interval:
            self._cells[cell_idx], self._cells[first_in_interval] = (
                self._cells[first_in_interval],
                self._cells[cell_idx],
            )

        # The modified cell is now at first_in_interval.
        # Shrink old_pop interval from the left.
        self._interval_begin[old_pop] += 1
        self._interval_size[old_pop] -= 1

        # Grow new_pop interval to the right (begin stays, size grows by 1).
        self._interval_size[new_pop] += 1
