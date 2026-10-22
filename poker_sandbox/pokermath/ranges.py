"""Hand ranges: ``"AKs, QQ+, T9o, 22-55, A5s-A2s"`` → weighted combos.

Parsing produces weight 1.0; ``Combo.weight`` exists so later modules can re-weight.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import combinations, product

from adapter.models import SUIT_CHARS, Card
from pokermath.cards import Rank, canonical

__all__ = ["TOTAL_COMBOS", "Combo", "Range"]

TOTAL_COMBOS = 1326


@dataclass(frozen=True, slots=True, order=True)
class Combo:
    """Two hole cards (stronger first) with a weight in [0, 1]."""

    c1: Card
    c2: Card
    weight: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        if self.c1 == self.c2:
            raise ValueError("combo cards must differ")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        hi, lo = canonical((self.c1, self.c2))
        object.__setattr__(self, "c1", hi)
        object.__setattr__(self, "c2", lo)

    @property
    def cards(self) -> tuple[Card, Card]:
        """``(c1, c2)``."""
        return self.c1, self.c2


def _pair_combos(rank: int) -> list[Combo]:
    return [Combo(Card(rank, a), Card(rank, b)) for a, b in combinations(SUIT_CHARS, 2)]


def _unpaired_combos(hi: int, lo: int, suited: bool | None) -> list[Combo]:
    out = []
    for s1, s2 in product(SUIT_CHARS, repeat=2):
        if suited is None or (s1 == s2) == suited:
            out.append(Combo(Card(hi, s1), Card(lo, s2)))
    return out


def _parse_hand(text: str) -> tuple[int, int, bool | None]:
    """``"AKs"`` → (14, 13, True); ``"QQ"`` → (12, 12, None)."""
    if len(text) not in (2, 3):
        raise ValueError(f"invalid hand {text!r}")
    r1, r2 = Rank.from_char(text[0]).value, Rank.from_char(text[1]).value
    suffix = text[2:].lower()
    if suffix not in ("", "s", "o"):
        raise ValueError(f"invalid suffix in {text!r}")
    suited = None if not suffix else suffix == "s"
    if r1 == r2 and suited is not None:
        raise ValueError(f"pairs take no s/o suffix: {text!r}")
    return max(r1, r2), min(r1, r2), suited


def _expand(token: str) -> list[Combo]:
    if "-" in token:
        left, right = (t.strip() for t in token.split("-", 1))
        a, b = _parse_hand(left), _parse_hand(right)
        if a[0] == a[1] and b[0] == b[1]:  # 22-55
            lo, hi = sorted((a[0], b[0]))
            return [c for r in range(lo, hi + 1) for c in _pair_combos(r)]
        if a[0] != b[0] or a[2] != b[2] or a[0] == a[1] or b[0] == b[1]:
            raise ValueError(f"span needs the same top card and suitedness: {token!r}")
        lo, hi = sorted((a[1], b[1]))  # A5s-A2s: kicker span
        return [c for k in range(lo, hi + 1) for c in _unpaired_combos(a[0], k, a[2])]
    plus = token.endswith("+")
    hi, lo, suited = _parse_hand(token.rstrip("+"))
    if hi == lo:
        return [c for r in range(hi, 15 if plus else hi + 1) for c in _pair_combos(r)]
    kickers = range(lo, hi) if plus else range(lo, lo + 1)  # ATs+ = ATs..AKs
    return [c for k in kickers for c in _unpaired_combos(hi, k, suited)]


@dataclass(frozen=True, slots=True)
class Range:
    """Immutable set of weighted combos (each card pair at most once)."""

    items: tuple[Combo, ...] = ()

    @classmethod
    def parse(cls, text: str) -> Range:
        """Parse comma-separated tokens; duplicates are merged."""
        seen: dict[tuple[Card, Card], Combo] = {}
        for token in (t.strip() for t in text.split(",")):
            if not token:
                continue
            for combo in _expand(token):
                seen[combo.cards] = combo
        if not seen:
            raise ValueError("empty range")
        return cls(tuple(sorted(seen.values(), reverse=True)))

    @classmethod
    def from_combos(cls, combos: Iterable[Combo]) -> Range:
        """Build from explicit combos (last one wins on duplicates)."""
        seen = {c.cards: c for c in combos}
        return cls(tuple(sorted(seen.values(), reverse=True)))

    def combos(self) -> list[tuple[Card, Card]]:
        """Card pairs, stronger card first."""
        return [c.cards for c in self.items]

    def weights(self) -> list[float]:
        """Weights aligned with :meth:`combos`."""
        return [c.weight for c in self.items]

    def remove_blockers(self, dead: Iterable[Card]) -> Range:
        """Drop combos that use any of ``dead``."""
        blocked = set(dead)
        return Range(tuple(c for c in self.items if c.c1 not in blocked and c.c2 not in blocked))

    def percent(self) -> float:
        """Weighted share of all 1326 starting hands, in percent (0..100)."""
        return 100.0 * sum(c.weight for c in self.items) / TOTAL_COMBOS

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Combo]:
        return iter(self.items)
