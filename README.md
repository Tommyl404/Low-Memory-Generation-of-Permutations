# littlebrain-rlcard

**Dealer algorithms from [arXiv:2505.01287](https://arxiv.org/abs/2505.01287) integrated with [RLCard](https://github.com/datamllab/rlcard).**

This repository implements four card-dealer algorithms — BitmapDealer, FisherYatesDealer, AdaptiveThresholdDealer, and PerfectDealer — with a uniform interface, and integrates them into RLCard as two custom environments:

| Environment ID | Description |
|---|---|
| `shuffle-guess` | Single-player card-guessing game measuring dealer predictability |
| `no-limit-holdem-dealerlab` | No-Limit Hold'em with selectable shuffle algorithm |

## Quickstart

```bash
# Install (editable, with dev dependencies)
python -m pip install -e ".[dev]"

# Run tests
pytest

# Run examples
python -m littlebrain_rlcard.scripts.run_examples

# Benchmark dealers
python -m littlebrain_rlcard.scripts.bench

# Sweep m_bits parameter
python -m littlebrain_rlcard.scripts.sweep_m_bits
```

## Dealer Algorithms

| Dealer | Paper Reference | Description |
|---|---|---|
| `BitmapDealer` | Algorithm 2.1 | Rejection sampling against availability bitmap |
| `FisherYatesDealer` | Algorithm 2.2 | Classic swap-delete (Knuth shuffle) |
| `AdaptiveThresholdDealer` | Algorithm 3.1 | Two-phase: adaptive threshold + swap-delete final |
| `PerfectDealer` | Appendix A, §4.1 | Cells/intervals/population structure |

All dealers share a common interface:

```python
from littlebrain_rlcard.dealers import get_dealer
from rlcard.utils.seeding import np_random

rng, _ = np_random(42)
dealer = get_dealer("adaptive")
dealer.reset(n=104, np_random=rng, m_bits=64)

cards = [dealer.draw() for _ in range(104)]
assert sorted(cards) == list(range(104))
```

## RLCard Environments

### shuffle-guess

```python
import rlcard
import littlebrain_rlcard  # registers envs

env = rlcard.make("shuffle-guess", config={
    "seed": 42,
    "game_dealer": "adaptive",
    "game_m_bits": 64,
    "game_n_cards": 104,
})
```

### no-limit-holdem-dealerlab

```python
env = rlcard.make("no-limit-holdem-dealerlab", config={
    "seed": 42,
    "game_dealer_algo": "fisher_yates",
    "game_num_players": 2,
})
```

## Example Output

```
=== Shuffle-Guess demo (adaptive, n=104, m_bits=64, seed=42) ===
Score (correct guesses): 3.0
First 10 drawn IDs: [37, 82, 5, ...]

=== First-card distribution (FisherYates, n=52, 1000 runs) ===
card_id  count
0        18
1        21
...
Chi2=49.2  df=51
```

## Project Structure

```
littlebrain-rlcard/
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
├── LICENSE
├── artifacts/plots/
├── src/littlebrain_rlcard/
│   ├── __init__.py
│   ├── cards.py
│   ├── rlcard_register.py
│   ├── dealers/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── bitmap.py
│   │   ├── fisher_yates.py
│   │   ├── adaptive_threshold.py
│   │   └── perfect.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── shuffle_guess/
│   │   │   ├── __init__.py
│   │   │   ├── game.py
│   │   │   └── env.py
│   │   └── holdem_dealerlab/
│   │       ├── __init__.py
│   │       ├── dealer.py
│   │       ├── game_limit.py
│   │       ├── game_nolimit.py
│   │       └── env_nolimit.py
│   └── scripts/
│       ├── __init__.py
│       ├── run_examples.py
│       ├── bench.py
│       ├── sweep_m_bits.py
│       └── plot_results.py
└── tests/
    ├── test_dealers_permutation.py
    ├── test_dealers_stats.py
    ├── test_rlcard_shuffle_guess_integration.py
    └── test_rlcard_holdem_integration.py
```

## Known Limitations

- **PerfectDealer PopSampler**: Uses a simple O(w) prefix-sum scan, suitable for n ≤ 256 (w ≤ 8). For larger n, Section 4.2 of the paper describes a constant-time dynamic sampler.
- **AdaptiveThresholdDealer `holes_elias_doc`**: The Elias-gamma encoding mode has the same runtime as `naive`; it adds docstrings/accounting for the theoretical bit cost but does not implement variable-length bit arrays.
- **Hold'em wrapper**: Single-deck only (52 cards) for evaluator compatibility.

## References

- Menuhin, S. & Naor, J. (2025). *Shuffling Cards When You Are of Very Little Brain: Low Memory Generation of Permutations*. [arXiv:2505.01287](https://arxiv.org/abs/2505.01287)
- Zha, D., et al. (2020). *RLCard: A Platform for Reinforcement Learning in Card Games*. [GitHub](https://github.com/datamllab/rlcard)

