"""revenue_tracker.py — P&L проекта за месяц: доходы (CPA + софт) − комиссии − расходы.

    python revenue_tracker.py --month 2026-01 --format json|table

Деньги считаются в Decimal, JSON читается с parse_float=Decimal — без ошибок двоичной плавающей
точки. Округление до цента (ROUND_HALF_UP) делается в одном месте на каждую величину:
комиссия CPA-сети — по каждому аккаунту (cpa_gross × cpa_commission), итоги — суммы уже
округлённых строк, поэтому сумма по таблице всегда равна итогу. В JSON суммы — строки «123.45».

Входные данные (форматы — договорённость этого модуля):
  data/cpa_stats.json      {"conversions": [{"id", "ts", "account_id", "offer_id", "payout_usd",
                                             "status": approved|pending|rejected}]}
                           status не указан → approved; pending идёт отдельной строкой, в доход не входит
  data/software_sales.json {"sales": [{"id", "ts", "account_id", "product", "amount_usd",
                                       "fees_usd": 0, "refunded": false}]}
  data/expenses.json       {"items": [{"month": "2026-01", "category": "...", "amount_usd", "note"}]}
                           category "gcp" — фактический счёт GCP: заменяет оценку budget_guard
  data/budget.json         оценка трат на Cloud Run (budget_guard): spent_usd текущего месяца
                           или history[месяц]
account_id — идентификатор из UTM-метки link_rotator (utm_campaign=account_<id>); в этом модуле
принято <device_id>_<platform>, например aria_fight_tiktok (так их считает CLI из devices.json).

Результат: data/analytics/pnl_<YYYY-MM>.json (atomically), его же рисует analytics-dashboard.js.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from diversity import atomic_write_text

log = logging.getLogger("revenue_tracker")

CENT = Decimal("0.01")
ZERO = Decimal("0")
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
OFFER_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
RECORD_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


def q(d: Decimal) -> Decimal:
    return d.quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(d: Decimal) -> str:
    return str(q(d))


def to_money(v: Any, what: str) -> Decimal:
    if isinstance(v, bool):
        raise ValueError(f"{what}: bool вместо суммы")
    try:
        d = v if isinstance(v, Decimal) else Decimal(str(v))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"{what}: не число ({v!r})") from e
    if not d.is_finite() or d < 0:
        raise ValueError(f"{what}: ожидалась неотрицательная сумма, получено {v!r}")
    return d


def validate_month(month: str) -> str:
    if not isinstance(month, str) or not MONTH_RE.fullmatch(month):
        raise ValueError(f"месяц должен быть YYYY-MM: {month!r}")
    return month


def month_of(ts: Any, tz_name: str = "UTC") -> str:
    if not isinstance(ts, str):
        raise ValueError(f"ts должен быть строкой ISO 8601: {ts!r}")
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    from zoneinfo import ZoneInfo
    return dt.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m")


def shift_month(month: str, n: int) -> str:
    y, m = map(int, month.split("-"))
    total = y * 12 + (m - 1) + n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def month_index(month: str) -> int:
    y, m = map(int, month.split("-"))
    return y * 12 + m - 1


# ---------------------------------------------------------------------------
# Разбор записей
# ---------------------------------------------------------------------------

@dataclass
class Quality:
    skipped: int = 0
    duplicates: int = 0
    pending_cpa: Decimal = ZERO
    refunds: Decimal = ZERO
    unknown_accounts: set[str] = field(default_factory=set)
    _warned: int = 0

    def skip(self, kind: str, rec: Any, why: str) -> None:
        self.skipped += 1
        if self._warned < 10:
            self._warned += 1
            log.warning("%s пропущена (%s): %r", kind, why, rec if not isinstance(rec, dict)
                        else {k: rec.get(k) for k in ("id", "account_id", "ts")})


def _offer_key(v: Any) -> str:
    if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
        return str(v)
    if isinstance(v, str) and OFFER_RE.fullmatch(v):
        return v
    raise ValueError(f"offer_id: {v!r}")


def _common(rec: Any, month: str, tz: str, seen: set[str]) -> tuple[str, str] | None:
    """(id, account_id) записи нужного месяца; None — запись другого месяца или дубль."""
    if not isinstance(rec, dict):
        raise ValueError("запись не объект")
    rid, acc = rec.get("id"), rec.get("account_id")
    if not isinstance(rid, str) or not RECORD_ID_RE.fullmatch(rid):
        raise ValueError(f"id: {rid!r}")
    if not isinstance(acc, str) or not ACCOUNT_RE.fullmatch(acc):
        raise ValueError(f"account_id: {acc!r}")
    if month_of(rec.get("ts"), tz) != month:
        return None
    if rid in seen:
        return ("", "")
    seen.add(rid)
    return rid, acc


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------

def compute_pnl(month: str, *, conversions: Iterable[Any], sales: Iterable[Any],
                expense_items: Iterable[Any], gcp_estimate_usd: Decimal | None, econ: Any,
                accounts: Iterable[str] = (), history: dict[str, Decimal] | None = None,
                tz: str = "UTC") -> dict:
    """Чистая функция: все входы переданы явно, ничего не читает с диска."""
    month = validate_month(month)
    rate = to_money(econ.cpa_commission, "cpa_commission")
    qa = Quality()

    cpa: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(lambda: ZERO))
    seen: set[str] = set()
    for rec in conversions:
        try:
            ids = _common(rec, month, tz, seen)
            if ids is None:
                continue
            if ids == ("", ""):
                qa.duplicates += 1
                continue
            status = rec.get("status", "approved")
            payout = to_money(rec.get("payout_usd"), "payout_usd")
            offer = _offer_key(rec.get("offer_id"))
            if status == "approved":
                cpa[ids[1]][offer] += payout
            elif status == "pending":
                qa.pending_cpa += payout
            elif status not in ("rejected", "reversed"):
                raise ValueError(f"status: {status!r}")
        except (ValueError, TypeError) as e:
            qa.skip("конверсия", rec, str(e))

    soft_gross: dict[str, Decimal] = defaultdict(lambda: ZERO)
    soft_fees: dict[str, Decimal] = defaultdict(lambda: ZERO)
    soft_count: dict[str, int] = defaultdict(int)
    seen = set()
    for rec in sales:
        try:
            ids = _common(rec, month, tz, seen)
            if ids is None:
                continue
            if ids == ("", ""):
                qa.duplicates += 1
                continue
            amount = to_money(rec.get("amount_usd"), "amount_usd")
            fees = to_money(rec.get("fees_usd", 0), "fees_usd")
            if rec.get("refunded") is True:
                qa.refunds += amount
                continue
            if fees > amount:
                raise ValueError("fees_usd больше amount_usd")
            soft_gross[ids[1]] += amount
            soft_fees[ids[1]] += fees
            soft_count[ids[1]] += 1
        except (ValueError, TypeError) as e:
            qa.skip("продажа", rec, str(e))

    known = set(accounts)
    qa.unknown_accounts = (set(cpa) | set(soft_gross)) - known if known else set()
    rows: list[dict[str, Any]] = []
    for acc in sorted(known | set(cpa) | set(soft_gross)):
        cpa_gross = q(sum(cpa[acc].values(), ZERO))
        commission = q(cpa_gross * rate)
        sg, sf = q(soft_gross[acc]), q(soft_fees[acc])
        gross = cpa_gross + sg
        rows.append({"account_id": acc,
                     "cpa_by_offer": {o: fmt(v) for o, v in sorted(cpa[acc].items())},
                     "cpa_gross": cpa_gross, "cpa_commission": commission,
                     "cpa_net": cpa_gross - commission, "software_sales": soft_count[acc],
                     "software_gross": sg, "software_fees": sf, "software_net": sg - sf,
                     "gross": gross, "net": gross - commission - sf})
    rows.sort(key=lambda r: (-r["net"], r["account_id"]))

    def total(key: str) -> Decimal:
        return sum((r[key] for r in rows), ZERO)

    # расходы
    other: dict[str, Decimal] = defaultdict(lambda: ZERO)
    gcp_invoice: Decimal | None = None
    for item in expense_items:
        try:
            if not isinstance(item, dict) or validate_month(str(item.get("month"))) != month:
                continue
            cat = str(item.get("category") or "other")
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", cat):
                raise ValueError(f"category: {cat!r}")
            amount = to_money(item.get("amount_usd"), "amount_usd")
            if cat == "gcp":
                gcp_invoice = (gcp_invoice or ZERO) + amount
            else:
                other[cat] += amount
        except (ValueError, TypeError) as e:
            qa.skip("расход", item, str(e))
    if gcp_invoice is not None:
        gcp, gcp_source = q(gcp_invoice), "expenses.json (счёт GCP)"
    elif gcp_estimate_usd is not None:
        gcp, gcp_source = q(gcp_estimate_usd), "budget_guard (оценка по времени GPU)"
    else:
        gcp, gcp_source = ZERO, "нет данных"
    other_q = {k: q(v) for k, v in sorted(other.items())}
    expenses_total = gcp + sum(other_q.values(), ZERO)

    gross_total = total("gross")
    direct = total("cpa_commission") + total("software_fees")
    net = gross_total - direct - expenses_total

    total_accounts = len(rows)
    active = sum(1 for r in rows if r["gross"] > 0)
    summary = {"total_accounts": total_accounts, "active_accounts": active,
               "revenue_per_account": fmt(gross_total / total_accounts) if total_accounts else "0.00",
               "arpu": fmt(gross_total / active) if active else "0.00",
               "net_per_account": fmt(net / total_accounts) if total_accounts else "0.00"}

    forecast, used = linear_forecast(month, net, history or {}, int(econ.forecast_months),
                                     int(econ.forecast_history_months))
    alerts = []
    if gcp > to_money(econ.gpu_cloud_threshold_usd, "gpu_cloud_threshold_usd"):
        alerts.append(f"облачный GPU ${fmt(gcp)} > порога ${econ.gpu_cloud_threshold_usd} в месяц — "
                      "пора считать переход на локальный GPU (gcp.mode=local)")
    if net < 0:
        alerts.append(f"месяц убыточный: ${fmt(net)}")
    if qa.unknown_accounts:
        alerts.append(f"доход у аккаунтов, которых нет в devices.json: {sorted(qa.unknown_accounts)}")

    return {
        "month": month, "currency": "USD",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accounts": [{k: (fmt(v) if isinstance(v, Decimal) else v) for k, v in r.items()} for r in rows],
        "totals": {k: fmt(total(k)) for k in ("cpa_gross", "cpa_commission", "cpa_net",
                                              "software_gross", "software_fees", "software_net",
                                              "gross", "net")},
        "expenses": {"gcp": {"amount": fmt(gcp), "source": gcp_source},
                     "other": {k: fmt(v) for k, v in other_q.items()}, "total": fmt(expenses_total)},
        "pnl": {"gross": fmt(gross_total), "direct_costs": fmt(direct),
                "expenses": fmt(expenses_total), "net": fmt(net)},
        "summary": summary,
        "forecast": {"method": "линейная регрессия по чистой прибыли", "history_months": used,
                     "months": forecast},
        "reinvest": reinvest_plan(net, econ),
        "alerts": alerts,
        "data_quality": {"skipped_records": qa.skipped, "duplicates": qa.duplicates,
                         "pending_cpa_usd": fmt(qa.pending_cpa), "refunds_usd": fmt(qa.refunds)},
    }


def linear_forecast(month: str, net: Decimal, history: dict[str, Decimal], ahead: int,
                    max_history: int) -> tuple[list[dict], list[str]]:
    """Прямая по методу наименьших квадратов через (номер месяца, чистая прибыль).
    Один месяц данных → прогноз равен ему (наклон не определён)."""
    past = sorted(m for m in history if MONTH_RE.fullmatch(m) and m < month)[-max_history:]
    points = [(month_index(m), Decimal(history[m])) for m in past] + [(month_index(month), net)]
    n = Decimal(len(points))
    mx = sum((Decimal(x) for x, _ in points), ZERO) / n
    my = sum((y for _, y in points), ZERO) / n
    sxx = sum(((Decimal(x) - mx) ** 2 for x, _ in points), ZERO)
    slope = (sum(((Decimal(x) - mx) * (y - my) for x, y in points), ZERO) / sxx) if sxx else ZERO
    out = []
    for k in range(1, ahead + 1):
        m = shift_month(month, k)
        out.append({"month": m, "net": fmt(my + slope * (Decimal(month_index(m)) - mx))})
    return out, past + [month]


def reinvest_plan(net: Decimal, econ: Any) -> dict:
    cost = to_money(econ.new_account_cost, "new_account_cost")
    cap = int(econ.accounts_per_month_cap)
    budget = max(net, ZERO)
    possible = int(budget // cost) if cost > 0 else 0
    launch = min(possible, cap)
    if possible > cap:
        text = f"можно запустить {possible} аккаунтов, но потолок {cap}/мес -> запускаем {launch}"
    else:
        text = f"можно запустить {possible} аккаунтов -> запускаем {launch}"
    return {"net_available": fmt(budget), "account_cost": fmt(cost), "possible": possible,
            "cap": cap, "launch": launch, "launch_cost": fmt(cost * launch),
            "left_after_launch": fmt(budget - cost * launch), "text": text}


# ---------------------------------------------------------------------------
# Чтение файлов
# ---------------------------------------------------------------------------

def read_json(path: Path, key: str) -> list:
    """Список записей из {"<key>": [...]} или просто [...]; нет файла — []."""
    if not path.exists():
        log.info("нет файла %s — считаю пустым", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    items = data.get(key) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: ожидался список или {{\"{key}\": [...]}}")
    return items


def gcp_estimate(budget_state: Path, month: str) -> Decimal | None:
    if not budget_state.exists():
        return None
    st = json.loads(budget_state.read_text(encoding="utf-8"), parse_float=Decimal)
    if st.get("month") == month:
        return to_money(st.get("spent_usd", 0), "budget.spent_usd")
    if month in (st.get("history") or {}):
        return to_money(st["history"][month], "budget.history")
    return None


def load_history(analytics_dir: Path, before: str) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for f in sorted(analytics_dir.glob("pnl_*.json")):
        m = f.stem[len("pnl_"):]
        if not MONTH_RE.fullmatch(m) or m >= before:
            continue
        try:
            out[m] = Decimal(json.loads(f.read_text(encoding="utf-8"))["pnl"]["net"])
        except Exception as e:
            log.warning("%s не читается: %s", f.name, e)
    return out


def accounts_from_registry(devices: Iterable[dict]) -> list[str]:
    """Те же id, что link_rotator ставит в utm_campaign (account_id_for)."""
    from link_rotator import account_id_for
    return sorted(account_id_for(d["device_id"], p) for d in devices
                  for p, a in (d.get("accounts") or {}).items() if a.get("enabled", True))


def build_month(cfg: Any, month: str, accounts: Iterable[str] = ()) -> tuple[dict, Path]:
    e = cfg.economics
    root = Path(cfg.root)

    def rel(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else root / pp

    analytics = cfg.path("analytics")
    result = compute_pnl(
        month, conversions=read_json(rel(e.cpa_stats), "conversions"),
        sales=read_json(rel(e.software_sales), "sales"),
        expense_items=read_json(rel(e.expenses), "items"),
        gcp_estimate_usd=gcp_estimate(cfg.budget_state_path, month), econ=e, accounts=accounts,
        history=load_history(analytics, month), tz=getattr(cfg, "timezone", "UTC"))
    out = analytics / f"pnl_{month}.json"
    atomic_write_text(out, json.dumps(result, ensure_ascii=False, indent=2))
    log.info("P&L %s: валовая $%s, чистыми $%s → %s", month, result["pnl"]["gross"],
             result["pnl"]["net"], out)
    return result, out


def render_table(r: dict) -> str:
    head = f"{'аккаунт':<28}{'CPA':>12}{'софт':>12}{'валовая':>12}{'комиссии':>12}{'чистыми':>12}"
    lines = [f"P&L {r['month']} (USD)", head, "-" * len(head)]
    for a in r["accounts"]:
        fees = Decimal(a["cpa_commission"]) + Decimal(a["software_fees"])
        lines.append(f"{a['account_id']:<28}{a['cpa_gross']:>12}{a['software_gross']:>12}"
                     f"{a['gross']:>12}{fmt(fees):>12}{a['net']:>12}")
    t = r["totals"]
    lines += ["-" * len(head),
              f"{'ИТОГО':<28}{t['cpa_gross']:>12}{t['software_gross']:>12}{t['gross']:>12}"
              f"{fmt(Decimal(t['cpa_commission']) + Decimal(t['software_fees'])):>12}{t['net']:>12}",
              "", f"Расходы: GCP ${r['expenses']['gcp']['amount']} ({r['expenses']['gcp']['source']})"]
    lines += [f"         {k} ${v}" for k, v in r["expenses"]["other"].items()]
    p, s = r["pnl"], r["summary"]
    lines += [f"Чистая прибыль: ${p['net']}  (валовая ${p['gross']} − комиссии ${p['direct_costs']}"
              f" − расходы ${p['expenses']})",
              f"Аккаунтов: {s['total_accounts']} (с доходом {s['active_accounts']}), доход на аккаунт "
              f"${s['revenue_per_account']}, ARPU ${s['arpu']}, чистыми на аккаунт ${s['net_per_account']}",
              "Прогноз: " + ", ".join(f"{f['month']} ${f['net']}" for f in r["forecast"]["months"]),
              "Реинвест: " + r["reinvest"]["text"]]
    lines += [f"! {a}" for a in r["alerts"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    from config_loader import cli_config

    def extra(ap):
        ap.add_argument("--month", default=None, help="YYYY-MM (по умолчанию текущий)")
        ap.add_argument("--format", choices=("json", "table"), default="table")
        ap.add_argument("--no-registry", action="store_true",
                        help="не брать список аккаунтов из devices.json")

    cfg, a = cli_config("revenue_tracker", extra=extra)
    month = validate_month(a.month or cfg.today()[:7])
    accounts: list[str] = []
    if not a.no_registry:
        from devices_registry import DevicesRegistry
        accounts = accounts_from_registry(DevicesRegistry.from_config(cfg).get_active_devices())
    result, _ = build_month(cfg, month, accounts)
    if a.format == "json":
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(render_table(result))


if __name__ == "__main__":
    main()
