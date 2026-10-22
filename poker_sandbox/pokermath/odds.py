"""Pot odds, required equity, implied odds and SPR. Pure functions; ``pot`` excludes hero's call."""

from __future__ import annotations

__all__ = [
    "break_even_future_win",
    "effective_stack",
    "implied_required_equity",
    "pot_odds",
    "required_equity",
    "spr",
]


def _check(pot: float, call: float) -> None:
    if pot < 0 or call < 0:
        raise ValueError("pot and call must be non-negative")
    if pot + call == 0:
        raise ValueError("pot + call must be positive")


def pot_odds(pot: float, call: float) -> float:
    """Pot-to-call ratio (``pot / call``), e.g. 2.0 means "2 to 1"."""
    _check(pot, call)
    if call == 0:
        raise ValueError("call must be positive for a ratio")
    return pot / call


def required_equity(pot: float, call: float) -> float:
    """Break-even equity for a call: ``call / (pot + call)``."""
    _check(pot, call)
    return call / (pot + call)


def implied_required_equity(pot: float, call: float, expected_future_win: float) -> float:
    """Break-even equity when hero also expects to win ``expected_future_win`` later.

    Heuristic: ``call / (pot + call + expected_future_win)``; ignores reverse implied odds.
    """
    if expected_future_win < 0:
        raise ValueError("expected_future_win must be non-negative")
    _check(pot, call)
    return call / (pot + call + expected_future_win)


def break_even_future_win(pot: float, call: float, equity: float) -> float:
    """Chips hero must win on later streets for a call with ``equity`` to break even (0 if already +EV)."""
    if not 0 < equity <= 1:
        raise ValueError("equity must be in (0, 1]")
    _check(pot, call)
    return max(0.0, call / equity - (pot + call))


def effective_stack(*stacks: float) -> float:
    """Most chips any player can lose: the second-largest stack (heads-up: the smaller one)."""
    if len(stacks) < 2:
        raise ValueError("need at least two stacks")
    if any(s < 0 for s in stacks):
        raise ValueError("stacks must be non-negative")
    return sorted(stacks, reverse=True)[1]


def spr(effective_stack: float, pot: float) -> float:
    """Stack-to-pot ratio: ``effective_stack / pot``."""
    if pot <= 0:
        raise ValueError("pot must be positive")
    if effective_stack < 0:
        raise ValueError("effective_stack must be non-negative")
    return effective_stack / pot
