"""Monte Carlo equity, vectorised with numpy.

Simulations run in fixed-size batches from one ``np.random.default_rng(seed)``,
so a given (inputs, n_simulations, seed) always yields the same numbers.
Each simulation scores 1 (win), 1/k (k-way split) or 0; equity is the mean and
the standard error is ``std(ddof=1) / sqrt(n)`` (95% CI = ±1.96·SE).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from adapter.models import Card
from pokermath.cards import to_index
from pokermath.hand_eval import evaluate_batch
from pokermath.ranges import Range

__all__ = ["BATCH_SIZE", "equity", "equity_vs_range", "equity_vs_range_with_ci", "equity_with_ci"]

BATCH_SIZE = 32_768
_DEAD_KEY = 2.0  # rng.random() is in [0, 1): dead cards never get picked


def _validate(hero: Sequence[Card], board: Sequence[Card]) -> tuple[list[int], list[int]]:
    if len(hero) != 2:
        raise ValueError("hero needs exactly 2 cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board must have 0, 3, 4 or 5 cards")
    hero_idx, board_idx = [to_index(c) for c in hero], [to_index(c) for c in board]
    if len(set(hero_idx + board_idx)) != len(hero_idx) + len(board_idx):
        raise ValueError("duplicate cards between hero and board")
    return hero_idx, board_idx


def _draw(rng: np.random.Generator, keys: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.int64]:
    """``k`` distinct live cards per row, in uniformly random order."""
    if k == 0:
        return np.empty((keys.shape[0], 0), dtype=np.int64)
    picked = np.argpartition(keys, k - 1, axis=1)[:, :k]
    # argpartition's output order depends on index layout; re-sort by key to keep the order random.
    order = np.argsort(np.take_along_axis(keys, picked, axis=1), axis=1)
    return np.take_along_axis(picked, order, axis=1).astype(np.int64)


def _score(hero_vals: npt.NDArray[np.int64], opp_vals: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Per-row share of the pot for hero against ``(N, n_opp)`` opponent values."""
    best = opp_vals.max(axis=1)
    ties = (opp_vals == hero_vals[:, None]).sum(axis=1)
    share = np.where(hero_vals > best, 1.0, np.where(hero_vals == best, 1.0 / (ties + 1), 0.0))
    return share.astype(np.float64)


def _summarise(total: float, total_sq: float, n: int) -> tuple[float, float]:
    mean = total / n
    if n < 2:
        return mean, 0.0
    var = max(0.0, (total_sq - n * mean * mean) / (n - 1))
    return mean, float(np.sqrt(var / n))


def _batches(n_simulations: int) -> list[int]:
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")
    full, rest = divmod(n_simulations, BATCH_SIZE)
    return [BATCH_SIZE] * full + ([rest] if rest else [])


def equity_with_ci(
    hero: Sequence[Card], board: Sequence[Card], n_opponents: int, n_simulations: int, seed: int
) -> tuple[float, float]:
    """Hero equity vs ``n_opponents`` random hands, with its standard error."""
    if not 1 <= n_opponents <= 9:
        raise ValueError("n_opponents must be in 1..9")
    hero_idx, board_idx = _validate(hero, board)
    rng = np.random.default_rng(seed)
    dead = hero_idx + board_idx
    live = np.setdiff1d(np.arange(52), dead)
    n_board = 5 - len(board_idx)
    need = n_board + 2 * n_opponents
    total = total_sq = 0.0
    for size in _batches(n_simulations):
        drawn = live[_draw(rng, rng.random((size, live.size)), need)]
        full_board = np.hstack([np.broadcast_to(np.array(board_idx, dtype=np.int64), (size, len(board_idx))),
                                drawn[:, :n_board]])
        hero_vals = evaluate_batch(np.hstack([np.broadcast_to(np.array(hero_idx), (size, 2)), full_board]))
        opp_vals = np.stack(
            [evaluate_batch(np.hstack([drawn[:, n_board + 2 * i : n_board + 2 * i + 2], full_board]))
             for i in range(n_opponents)],
            axis=1,
        )
        share = _score(hero_vals, opp_vals)
        total += float(share.sum())
        total_sq += float((share * share).sum())
    return _summarise(total, total_sq, n_simulations)


def equity(hero: Sequence[Card], board: Sequence[Card], n_opponents: int, n_simulations: int, seed: int) -> float:
    """Hero equity vs ``n_opponents`` uniformly random hands."""
    return equity_with_ci(hero, board, n_opponents, n_simulations, seed)[0]


def equity_vs_range_with_ci(
    hero: Sequence[Card], board: Sequence[Card], villain_range: Range, n_simulations: int, seed: int
) -> tuple[float, float]:
    """Hero equity vs one villain drawn from ``villain_range`` by weight, with its standard error."""
    hero_idx, board_idx = _validate(hero, board)
    live_range = villain_range.remove_blockers([*hero, *board])
    weights = np.array(live_range.weights(), dtype=np.float64)
    if weights.size == 0 or weights.sum() <= 0:
        raise ValueError("villain range is empty after removing blockers")
    probs = weights / weights.sum()
    combos = np.array([[to_index(a), to_index(b)] for a, b in live_range.combos()], dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_board = 5 - len(board_idx)
    board_arr = np.array(board_idx, dtype=np.int64)
    total = total_sq = 0.0
    for size in _batches(n_simulations):
        villain = combos[rng.choice(combos.shape[0], size=size, p=probs)]
        keys = rng.random((size, 52))
        keys[:, hero_idx + board_idx] = _DEAD_KEY
        rows = np.arange(size)[:, None]
        keys[rows, villain] = _DEAD_KEY  # per-row dead cards
        full_board = np.hstack([np.broadcast_to(board_arr, (size, board_arr.size)), _draw(rng, keys, n_board)])
        hero_vals = evaluate_batch(np.hstack([np.broadcast_to(np.array(hero_idx), (size, 2)), full_board]))
        opp_vals = evaluate_batch(np.hstack([villain, full_board]))[:, None]
        share = _score(hero_vals, opp_vals)
        total += float(share.sum())
        total_sq += float((share * share).sum())
    return _summarise(total, total_sq, n_simulations)


def equity_vs_range(
    hero: Sequence[Card], board: Sequence[Card], villain_range: Range, n_simulations: int, seed: int
) -> float:
    """Hero equity vs one villain drawn from ``villain_range`` by weight."""
    return equity_vs_range_with_ci(hero, board, villain_range, n_simulations, seed)[0]
