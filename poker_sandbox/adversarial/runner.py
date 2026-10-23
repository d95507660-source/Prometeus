"""Play one :class:`AdversarialConfig` on the offline M0 mock and collect what a detector would see.

``collusion.runner.run_scenario`` builds its strategies internally and cannot wrap them with
M5, so the same composition is rebuilt here from the public M3/M4/M5 pieces:
``HeuristicStrategy → CoordinationPolicy (group only) → HumanLikeWrapper``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adapter.config import AdapterConfig, Stakes
from adapter.mock import MockAdapter
from adapter.models import Action, HandResult, TableState
from adversarial.space import AdversarialConfig
from agent.strategy import Strategy
from behavior.humanlike import HumanLikeWrapper
from collusion.channel import InProcessChannel
from collusion.policy import CoordinationPolicy
from collusion.runner import legality_error
from strategy.heuristic import HeuristicStrategy
from strategy.params import HeuristicParams
from strategy.sim import play_hands

__all__ = ["RunOutput", "RunSettings", "run_config"]


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Everything besides the config that shapes a run (part of the cache key).

    Attributes:
        n_hands: Hands per evaluation.
        seed: Deck and bot seed, shared by every config so comparisons are paired.
        n_seats: Table size (target seats are 0..group_size-1, the rest is a fixed field of default bots).
        equity_samples: Monte Carlo samples per postflop decision (speed knob).
    """

    n_hands: int = 300
    seed: int = 1
    n_seats: int = 6
    equity_samples: int = 200

    def __post_init__(self) -> None:
        if self.n_hands < 1:
            raise ValueError("n_hands must be >= 1")
        if not 2 <= self.n_seats <= 6:
            raise ValueError("n_seats must be 2..6")
        if self.equity_samples < 50:
            raise ValueError("equity_samples must be >= 50")

    def as_dict(self) -> dict[str, int]:
        """Plain dict for logs and cache keys."""
        return {"n_hands": self.n_hands, "seed": self.seed, "n_seats": self.n_seats,
                "equity_samples": self.equity_samples}


@dataclass(slots=True)
class RunOutput:
    """Inputs for :meth:`DetectorOracle.score` plus run counters."""

    hand_history: list[HandResult]
    agent_meta: dict[str, Any]
    behavior_log: list[dict[str, Any]]
    adjustments: int = 0


def _strict(seat: int, state: TableState, action: Action) -> None:
    if (error := legality_error(seat, state, action)) is not None:
        raise AssertionError(f"illegal decision at {state.hand_id}: {error}")


def run_config(config: AdversarialConfig, settings: RunSettings) -> RunOutput:
    """Play ``settings.n_hands`` hands; seats ``0..group_size-1`` use ``config``, the others default M3 bots."""
    if config.group_size > settings.n_seats - 1:
        raise ValueError("group must leave at least one field seat")
    seed, stakes = settings.seed, Stakes()
    targets = tuple(range(config.group_size))
    mock = MockAdapter()
    mock.connect(AdapterConfig(seats=settings.n_seats, seed=seed, auto_deal=False, hero_seat=None,
                               stakes=stakes, table_id="adversarial"))
    strategies: dict[int, Strategy] = {}
    for seat in range(settings.n_seats):  # bot seeds depend on the seat only: the field is identical across configs
        params = config.heuristic_params(seed * 1000 + seat, settings.equity_samples) if seat in targets else \
            HeuristicParams(equity_samples=settings.equity_samples, seed=seed * 1000 + seat)
        strategies[seat] = HeuristicStrategy(params)
    channel = InProcessChannel(targets) if config.collusion else None
    policies: list[CoordinationPolicy] = []
    if channel is not None:
        for seat in targets:
            policy = CoordinationPolicy(strategies[seat], seat, frozenset(targets), channel,
                                        config.coordination_params(seed * 7919 + seat))
            policies.append(policy)
            strategies[seat] = policy
    wrappers = {seat: HumanLikeWrapper(strategies[seat], config.humanlike_profile(settings.n_hands),
                                       seed=seed * 104729 + seat, seat=seat) for seat in targets}
    strategies.update(wrappers)
    try:
        results = play_hands(mock, strategies, settings.n_hands, stakes.starting_stack, _strict)
    finally:
        for policy in policies:
            policy.close()
        if channel is not None:
            channel.close()
        for w in wrappers.values():
            w.close()
        mock.disconnect()
    behavior_log = [r.as_log() for w in wrappers.values() for r in w.records]  # no ts: deterministic
    meta: dict[str, Any] = {
        "target_seats": list(targets),  # seats the detector is asked about (a suspected group if > 1)
        "n_seats": settings.n_seats,
        "big_blind": stakes.big_blind,
        "n_hands": settings.n_hands,
        "strategy_name": wrappers[0].name,
    }
    return RunOutput(results, meta, behavior_log, adjustments=sum(p.adjustments_total for p in policies))
