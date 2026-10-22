"""Poker math core (M2): cards, hand evaluation, Monte Carlo equity, ranges, odds.

Independent of the adapter runtime: only ``Card`` is imported from ``adapter.models``.
"""

from pokermath.cards import (
    Card,
    Rank,
    Suit,
    canonical,
    from_index,
    full_deck_indices,
    parse,
    parse_many,
    to_index,
)
from pokermath.equity import BATCH_SIZE, equity, equity_vs_range, equity_vs_range_with_ci, equity_with_ci
from pokermath.hand_eval import CATEGORY_NAMES, HandValue, best_five, compare, evaluate, evaluate_batch
from pokermath.odds import (
    break_even_future_win,
    effective_stack,
    implied_required_equity,
    pot_odds,
    required_equity,
    spr,
)
from pokermath.ranges import Combo, Range

__all__ = [
    "BATCH_SIZE", "CATEGORY_NAMES", "Card", "Combo", "HandValue", "Range", "Rank", "Suit", "best_five",
    "break_even_future_win", "canonical", "compare", "effective_stack", "equity", "equity_vs_range",
    "equity_vs_range_with_ci", "equity_with_ci", "evaluate", "evaluate_batch", "from_index",
    "full_deck_indices", "implied_required_equity", "parse", "parse_many", "pot_odds", "required_equity",
    "spr", "to_index",
]
