"""
Dealer algorithm implementations.

All four dealers share the :class:`~littlebrain_rlcard.dealers.common.BaseDealer`
interface and are accessible from this package:

>>> from littlebrain_rlcard.dealers import get_dealer
>>> dealer = get_dealer("fisher_yates")
"""

from __future__ import annotations

from littlebrain_rlcard.dealers.adaptive_threshold import AdaptiveThresholdDealer
from littlebrain_rlcard.dealers.bitmap import BitmapDealer
from littlebrain_rlcard.dealers.common import BaseDealer
from littlebrain_rlcard.dealers.fisher_yates import FisherYatesDealer
from littlebrain_rlcard.dealers.perfect import PerfectDealer

DEALER_REGISTRY: dict[str, type[BaseDealer]] = {
    "bitmap": BitmapDealer,
    "fisher_yates": FisherYatesDealer,
    "adaptive": AdaptiveThresholdDealer,
    "perfect": PerfectDealer,
}


def get_dealer(name: str) -> BaseDealer:
    """Instantiate a dealer by short name.

    Valid names: ``"bitmap"``, ``"fisher_yates"``, ``"adaptive"``, ``"perfect"``.
    """
    cls = DEALER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dealer {name!r}. Choose from {list(DEALER_REGISTRY)}"
        )
    return cls()


__all__ = [
    "BaseDealer",
    "BitmapDealer",
    "FisherYatesDealer",
    "AdaptiveThresholdDealer",
    "PerfectDealer",
    "DEALER_REGISTRY",
    "get_dealer",
]
