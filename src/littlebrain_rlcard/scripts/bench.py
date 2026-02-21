"""
bench.py — Benchmark deals/sec and memory for each dealer on n=104.

Usage::

    python -m littlebrain_rlcard.scripts.bench
"""

from __future__ import annotations

import time

from rlcard.utils.seeding import np_random as _np_random

from littlebrain_rlcard.dealers import DEALER_REGISTRY, get_dealer
from littlebrain_rlcard.dealers.common import _deep_getsizeof


def bench_dealer(name: str, n: int = 104, repeats: int = 200) -> dict:
    """Benchmark a single dealer.

    Returns dict with deals_per_sec, total_time, theoretical_bits,
    python_bytes.
    """
    dealer = get_dealer(name)
    rng, _ = _np_random(42)
    params = {"m_bits": 64} if name == "adaptive" else {}

    # Warm-up
    dealer.reset(n, rng, **params)
    for _ in range(n):
        dealer.draw()

    # Timed runs
    t0 = time.perf_counter()
    for r in range(repeats):
        rng2, _ = _np_random(r)
        dealer.reset(n, rng2, **params)
        for _ in range(n):
            dealer.draw()
    elapsed = time.perf_counter() - t0

    total_draws = repeats * n
    deals_per_sec = total_draws / elapsed

    # Memory snapshot (after last run, exhausted state)
    rng3, _ = _np_random(9999)
    dealer.reset(n, rng3, **params)
    summary = dealer.state_summary()
    py_bytes = _deep_getsizeof(dealer)

    return {
        "name": name,
        "n": n,
        "repeats": repeats,
        "total_draws": total_draws,
        "elapsed_s": elapsed,
        "deals_per_sec": deals_per_sec,
        "theoretical_bits": summary["theoretical_bits"],
        "python_bytes": py_bytes,
    }


def main() -> None:
    n = 104
    repeats = 200
    print(f"Benchmarking dealers on n={n}, {repeats} full permutations each\n")
    header = f"{'Dealer':<22} {'Draws/s':>12} {'Time (s)':>10} {'Theory bits':>12} {'Py bytes':>10}"
    print(header)
    print("-" * len(header))
    for name in DEALER_REGISTRY:
        res = bench_dealer(name, n=n, repeats=repeats)
        print(
            f"{res['name']:<22} {res['deals_per_sec']:>12,.0f} "
            f"{res['elapsed_s']:>10.3f} "
            f"{res['theoretical_bits']:>12} {res['python_bytes']:>10}"
        )
    print()


if __name__ == "__main__":
    main()
