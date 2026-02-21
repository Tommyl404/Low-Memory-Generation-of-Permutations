"""
Card-ID utilities for one- and two-deck representations.

Card IDs
--------
* ``card_id`` ∈ [0 .. 103] for two-deck mode (default), [0 .. 51] for single.
* ``deck_index = card_id // 52``   (0 or 1)
* ``type_id   = card_id % 52``     (suit × 13 + rank)

Two decks are used by the shuffle-guess environment (n=104).
Texas Hold'em stays single-deck (n=52) for evaluator compatibility.
"""

from __future__ import annotations

SUITS = ["S", "H", "D", "C"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

NUM_TYPES = 52
SINGLE_DECK = 52
DOUBLE_DECK = 104


def deck_index(card_id: int) -> int:
    """Return 0 for first deck, 1 for second deck."""
    return card_id // NUM_TYPES


def type_id(card_id: int) -> int:
    """Return the card type in [0..51]."""
    return card_id % NUM_TYPES


def suit_of(type_id_val: int) -> str:
    """Return suit character for a type_id."""
    return SUITS[type_id_val // 13]


def rank_of(type_id_val: int) -> str:
    """Return rank character for a type_id."""
    return RANKS[type_id_val % 13]


def pretty(card_id: int) -> str:
    """Human-readable card name, e.g. ``'AS(d0)'``."""
    tid = type_id(card_id)
    di = deck_index(card_id)
    return f"{rank_of(tid)}{suit_of(tid)}(d{di})"
