"""tests/test_pnl.py — P&L до цента, прогноз, реинвест-план, расходы, панель на Node.

Запуск:  python -m pytest -q tests/test_pnl.py
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_loader  # noqa: E402
import revenue_tracker as rt  # noqa: E402

MONTH = "2026-01"
D = Decimal


def cents(x: Decimal) -> Decimal:
    return x.quantize(D("0.01"), rounding=ROUND_HALF_UP)


def econ(**kw) -> config_loader.EconomicsConfig:
    return config_loader.EconomicsConfig(**kw)


def pnl(conversions=(), sales=(), expenses=(), gcp=None, accounts=(), history=None, tz="UTC", **econ_kw):
    return rt.compute_pnl(MONTH, conversions=conversions, sales=sales, expense_items=expenses,
                          gcp_estimate_usd=gcp, econ=econ(**econ_kw), accounts=accounts,
                          history=history, tz=tz)


# ---------------------------------------------------------------------------
# Моки: 30 аккаунтов, 100 конверсий, 20 продаж
# ---------------------------------------------------------------------------

DEVICES = [f"dev{i:02d}" for i in range(10)]
ACCOUNTS = [f"{d}_{p}" for d in DEVICES for p in ("tiktok", "instagram", "youtube")]


def mock_data(seed: int = 7):
    rnd = random.Random(seed)
    conversions = []
    for i in range(100):
        status = "approved" if i < 85 else ("pending" if i < 95 else "rejected")
        conversions.append({"id": f"c{i}", "ts": f"{MONTH}-{rnd.randint(1, 28):02d}T{rnd.randint(0, 23):02d}:15:00Z",
                            "account_id": rnd.choice(ACCOUNTS[:24]), "offer_id": rnd.choice([1, 2, 3, "promo_x"]),
                            "payout_usd": D(f"{rnd.randint(50, 900)}.{rnd.randint(0, 99):02d}") / 100 * 10,
                            "status": status})
    sales = []
    for i in range(20):
        amount = D(f"{rnd.randint(9, 49)}.{rnd.choice([0, 49, 99]):02d}")
        sales.append({"id": f"s{i}", "ts": f"{MONTH}-{rnd.randint(1, 28):02d}T12:00:00Z",
                      "account_id": rnd.choice(ACCOUNTS[20:]), "product": "habit_tracker",
                      "amount_usd": amount, "fees_usd": cents(amount * D("0.029") + D("0.30")),
                      "refunded": i in (3, 11)})
    return conversions, sales


def manual(conversions, sales, commission: Decimal, expenses_total: Decimal):
    """Независимый ручной расчёт по правилам из docstring revenue_tracker."""
    cpa = defaultdict(lambda: D(0))
    soft = defaultdict(lambda: D(0))
    fees = defaultdict(lambda: D(0))
    for c in conversions:
        if c["status"] == "approved":
            cpa[c["account_id"]] += c["payout_usd"]
    for s in sales:
        if not s["refunded"]:
            soft[s["account_id"]] += s["amount_usd"]
            fees[s["account_id"]] += s["fees_usd"]
    rows = {}
    for acc in ACCOUNTS:
        g = cents(cpa[acc])
        com = cents(g * commission)
        rows[acc] = {"cpa_gross": g, "cpa_commission": com, "software_gross": cents(soft[acc]),
                     "software_fees": cents(fees[acc]),
                     "net": g - com + cents(soft[acc]) - cents(fees[acc])}
    gross = sum((r["cpa_gross"] + r["software_gross"] for r in rows.values()), D(0))
    direct = sum((r["cpa_commission"] + r["software_fees"] for r in rows.values()), D(0))
    return rows, gross, direct, gross - direct - expenses_total


@pytest.mark.parametrize("commission", ["0", "0.1", "0.175"])
def test_pnl_matches_manual_to_the_cent(commission):
    conversions, sales = mock_data()
    expenses = [{"month": MONTH, "category": "cloud_phones", "amount_usd": D("120.00")},
                {"month": MONTH, "category": "llm_api", "amount_usd": D("37.45")},
                {"month": "2025-12", "category": "cloud_phones", "amount_usd": D("999")}]
    r = pnl(conversions, sales, expenses, gcp=D("212.336"), accounts=ACCOUNTS,
            cpa_commission=D(commission))
    exp_total = cents(D("212.336")) + D("120.00") + D("37.45")
    rows, gross, direct, net = manual(conversions, sales, D(commission), exp_total)

    assert r["summary"]["total_accounts"] == 30
    by_acc = {a["account_id"]: a for a in r["accounts"]}
    for acc, m in rows.items():
        for k, v in m.items():
            assert D(by_acc[acc][k]) == v, (acc, k)
    assert D(r["pnl"]["gross"]) == gross
    assert D(r["pnl"]["direct_costs"]) == direct
    assert D(r["expenses"]["total"]) == exp_total
    assert D(r["pnl"]["net"]) == net
    # сумма строк таблицы = итог
    assert sum(D(a["net"]) for a in r["accounts"]) == D(r["totals"]["net"])
    assert D(r["totals"]["net"]) - D(r["expenses"]["total"]) == D(r["pnl"]["net"])
    dq = r["data_quality"]
    pending = sum((c["payout_usd"] for c in conversions if c["status"] == "pending"), D(0))
    refunds = sum((s["amount_usd"] for s in sales if s["refunded"]), D(0))
    assert D(dq["pending_cpa_usd"]) == cents(pending) and D(dq["refunds_usd"]) == cents(refunds)


def test_small_case_with_literal_numbers():
    """Посчитано вручную: CPA a: 2.50+3.33+27.50 = 33.33; комиссия 15 % = 4.9995 → 5.00.
    Софт a: 19.99 − сборы 0.88 = 19.11. Чистыми a: 33.33 − 5.00 + 19.11 = 47.44.
    CPA b: 0.10+0.10+0.10 = 0.30 (а не 0.30000000000000004); комиссия 0.045 → 0.05; b = 0.25.
    Расходы: GCP 10.10 + прочие 5.55 = 15.65. Итог: 47.69 − 15.65 = 32.04."""
    ts = f"{MONTH}-10T10:00:00Z"
    conv = [{"id": "1", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": D("2.50")},
            {"id": "2", "ts": ts, "account_id": "a", "offer_id": 2, "payout_usd": D("3.33")},
            {"id": "3", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": D("27.50")},
            {"id": "4", "ts": ts, "account_id": "b", "offer_id": 1, "payout_usd": 0.1},
            {"id": "5", "ts": ts, "account_id": "b", "offer_id": 1, "payout_usd": 0.1},
            {"id": "6", "ts": ts, "account_id": "b", "offer_id": 1, "payout_usd": 0.1}]
    sales = [{"id": "s1", "ts": ts, "account_id": "a", "amount_usd": D("19.99"), "fees_usd": D("0.88")}]
    r = pnl(conv, sales, [{"month": MONTH, "category": "misc", "amount_usd": D("5.55")}],
            gcp=D("10.10"), cpa_commission=D("0.15"))
    a, b = r["accounts"]
    assert (a["account_id"], a["cpa_gross"], a["cpa_commission"], a["software_net"], a["net"]) == \
        ("a", "33.33", "5.00", "19.11", "47.44")
    assert a["cpa_by_offer"] == {"1": "30.00", "2": "3.33"}
    assert (b["cpa_gross"], b["cpa_commission"], b["net"]) == ("0.30", "0.05", "0.25")
    assert r["pnl"] == {"gross": "53.62", "direct_costs": "5.93", "expenses": "15.65", "net": "32.04"}
    assert r["summary"] == {"total_accounts": 2, "active_accounts": 2, "revenue_per_account": "26.81",
                            "arpu": "26.81", "net_per_account": "16.02"}


# ---------------------------------------------------------------------------
# Реинвест и прогноз
# ---------------------------------------------------------------------------

def test_reinvest_plan_cap():
    plan = rt.reinvest_plan(D("5000"), econ(new_account_cost=40, accounts_per_month_cap=8))
    assert plan["text"] == "можно запустить 125 аккаунтов, но потолок 8/мес -> запускаем 8"
    assert (plan["possible"], plan["launch"], plan["launch_cost"], plan["left_after_launch"]) == \
        (125, 8, "320.00", "4680.00")


@pytest.mark.parametrize("net, possible, launch", [("199.99", 4, 4), ("0", 0, 0), ("-50", 0, 0), ("40", 1, 1)])
def test_reinvest_plan_small_and_negative(net, possible, launch):
    plan = rt.reinvest_plan(D(net), econ())
    assert (plan["possible"], plan["launch"]) == (possible, launch)
    assert plan["text"] == f"можно запустить {possible} аккаунтов -> запускаем {launch}"


def test_linear_forecast():
    f, used = rt.linear_forecast(MONTH, D("3000"), {"2025-11": D("1000"), "2025-12": D("2000")}, 3, 6)
    assert [x["net"] for x in f] == ["4000.00", "5000.00", "6000.00"]
    assert [x["month"] for x in f] == ["2026-02", "2026-03", "2026-04"]
    assert used == ["2025-11", "2025-12", "2026-01"]


def test_forecast_single_month_is_flat_and_gaps_use_month_index():
    f, _ = rt.linear_forecast(MONTH, D("750.10"), {}, 2, 6)
    assert [x["net"] for x in f] == ["750.10", "750.10"]
    # пропущенный месяц: 2025-10 → 0, 2026-01 → 300 ⇒ +100/мес
    f, _ = rt.linear_forecast(MONTH, D("300"), {"2025-10": D("0")}, 1, 6)
    assert f[0]["net"] == "400.00"


def test_forecast_history_limit():
    hist = {rt.shift_month(MONTH, -k): D("100") for k in range(1, 13)}
    _, used = rt.linear_forecast(MONTH, D("100"), hist, 1, 6)
    assert len(used) == 7 and used[0] == "2025-07"


# ---------------------------------------------------------------------------
# Месяцы, качество данных, расходы
# ---------------------------------------------------------------------------

def test_month_boundary_respects_timezone():
    conv = [{"id": "x", "ts": "2026-01-31T23:30:00Z", "account_id": "a", "offer_id": 1, "payout_usd": 5}]
    assert pnl(conv)["pnl"]["gross"] == "5.00"
    assert pnl(conv, tz="Asia/Tokyo")["pnl"]["gross"] == "0.00"  # в Токио это уже февраль


def test_bad_and_duplicate_records_are_skipped():
    ts = f"{MONTH}-05T00:00:00Z"
    conv = [{"id": "1", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": 10},
            {"id": "1", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": 10},   # дубль
            {"id": "2", "ts": ts, "account_id": "a b", "offer_id": 1, "payout_usd": 10},  # плохой id
            {"id": "3", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": -5},   # минус
            {"id": "4", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": "abc"},
            {"id": "5", "ts": "вчера", "account_id": "a", "offer_id": 1, "payout_usd": 1},
            {"id": "6", "ts": ts, "account_id": "a", "offer_id": 1, "payout_usd": 1, "status": "weird"},
            "не объект"]
    sales = [{"id": "s", "ts": ts, "account_id": "a", "amount_usd": 5, "fees_usd": 9}]
    r = pnl(conv, sales)
    assert r["pnl"]["gross"] == "10.00"
    assert r["data_quality"]["duplicates"] == 1 and r["data_quality"]["skipped_records"] == 7


def test_gcp_invoice_overrides_estimate_and_threshold_alert():
    r = pnl(expenses=[{"month": MONTH, "category": "gcp", "amount_usd": D("350.00")}], gcp=D("280"))
    assert r["expenses"]["gcp"] == {"amount": "350.00", "source": "expenses.json (счёт GCP)"}
    assert any("пора считать переход на локальный GPU" in a for a in r["alerts"])
    r = pnl(gcp=D("280"))
    assert r["expenses"]["gcp"]["source"].startswith("budget_guard")
    assert not any("локальный GPU" in a for a in r["alerts"])


def test_unknown_accounts_flagged():
    ts = f"{MONTH}-05T00:00:00Z"
    r = pnl([{"id": "1", "ts": ts, "account_id": "ghost_tiktok", "offer_id": 1, "payout_usd": 3}],
            accounts=["dev00_tiktok"])
    assert r["summary"]["total_accounts"] == 2 and r["summary"]["active_accounts"] == 1
    assert any("ghost_tiktok" in a for a in r["alerts"])


def test_gcp_estimate_from_budget_state(tmp_path):
    f = tmp_path / "budget.json"
    f.write_text(json.dumps({"month": "2026-02", "spent_usd": 12.345678,
                             "history": {"2026-01": 280.1}}), encoding="utf-8")
    assert rt.gcp_estimate(f, "2026-02") == D("12.345678")
    assert rt.gcp_estimate(f, "2026-01") == D("280.1")
    assert rt.gcp_estimate(f, "2025-12") is None
    assert rt.gcp_estimate(tmp_path / "none.json", MONTH) is None


@pytest.mark.parametrize("month", ["2026-13", "2026-1", "26-01", "../x"])
def test_bad_month(month):
    with pytest.raises(ValueError):
        rt.validate_month(month)


# ---------------------------------------------------------------------------
# Файлы, история, CLI-таблица, конфиг
# ---------------------------------------------------------------------------

class FakeCfg:
    def __init__(self, root: Path):
        self.root = root
        self.timezone = "UTC"
        self.economics = econ(cpa_commission=0.1)

    def path(self, key):
        return self.root / {"analytics": "data/analytics"}[key]

    @property
    def budget_state_path(self):
        return self.root / "data" / "budget.json"


def _jdump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str), encoding="utf-8")


def test_build_month_writes_file_and_uses_history(tmp_path):
    cfg = FakeCfg(tmp_path)
    conversions, sales = mock_data()
    _jdump(tmp_path / "data" / "cpa_stats.json", {"conversions": conversions})
    _jdump(tmp_path / "data" / "software_sales.json", {"sales": sales})
    _jdump(tmp_path / "data" / "expenses.json", {"items": [{"month": MONTH, "category": "phones",
                                                            "amount_usd": "100.00"}]})
    _jdump(tmp_path / "data" / "budget.json", {"month": MONTH, "spent_usd": 50.0})
    _jdump(tmp_path / "data" / "analytics" / "pnl_2025-12.json", {"pnl": {"net": "100.00"}})
    r, out = rt.build_month(cfg, MONTH, ACCOUNTS)
    assert out == tmp_path / "data" / "analytics" / f"pnl_{MONTH}.json"
    assert json.loads(out.read_text(encoding="utf-8"))["pnl"] == r["pnl"]
    assert r["forecast"]["history_months"] == ["2025-12", MONTH]
    assert r["expenses"]["total"] == "150.00"
    table = rt.render_table(r)
    assert "ИТОГО" in table and r["reinvest"]["text"] in table and r["pnl"]["net"] in table


def test_accounts_from_registry():
    devs = [{"device_id": "aria_fight", "accounts": {"tiktok": {}, "instagram": {"enabled": False},
                                                     "youtube": {"enabled": True}}}]
    assert rt.accounts_from_registry(devs) == ["aria_fight_tiktok", "aria_fight_youtube"]


def _write_cfg(tmp_path: Path, mutate=None) -> Path:
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(raw)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    return tmp_path / "config.json"


def test_config_economics_loaded(tmp_path):
    e = config_loader.load_config(_write_cfg(tmp_path)).economics
    assert (e.cpa_commission, e.new_account_cost, e.accounts_per_month_cap, e.gpu_cloud_threshold_usd) == \
        (0.0, 40, 8, 300)


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["economics"].update(cpa_commission=1.5), "cpa_commission"),
    (lambda r: r["economics"].update(new_account_cost=0), "new_account_cost"),
    (lambda r: r["economics"].update(accounts_per_month_cap=-1), "accounts_per_month_cap"),
])
def test_config_economics_validation(tmp_path, mutate, msg):
    with pytest.raises(ValueError, match=msg):
        config_loader.load_config(_write_cfg(tmp_path, mutate))


# ---------------------------------------------------------------------------
# Панель на Node
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_dashboard_renders_python_numbers(tmp_path):
    cfg = FakeCfg(tmp_path)
    conversions, sales = mock_data()
    _jdump(tmp_path / "data" / "cpa_stats.json", {"conversions": conversions})
    _jdump(tmp_path / "data" / "software_sales.json", {"sales": sales})
    _jdump(tmp_path / "data" / "analytics" / "pnl_2025-12.json",
           {"month": "2025-12", "pnl": {"gross": "10.00", "direct_costs": "0.00", "expenses": "0.00",
                                        "net": "10.00"}})
    r, _ = rt.build_month(cfg, MONTH, ACCOUNTS)
    script = (f"const d=require({json.dumps(str(ROOT / 'analytics-dashboard.js'))});"
              f"d.buildDashboard({{project:'<P&L>'}},{{dataDir:{json.dumps(str(tmp_path / 'data'))},"
              "includeFans:false}).then(p=>process.stdout.write(p)).catch(e=>{console.error(e);process.exit(1)});"
              "process.stderr.write(JSON.stringify([d.usd('-1234.50'),d.usd('5000.00'),d.usd('0.30')]))")
    p = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stderr.splitlines()[0]) == ["−$1,234.50", "$5,000.00", "$0.30"]
    html = Path(p.stdout.strip().splitlines()[-1]).read_text(encoding="utf-8")
    assert "&lt;P&amp;L&gt;" in html and "<P&L>" not in html  # экранирование
    net = r["pnl"]["net"]
    int_part, frac = net.lstrip("-").split(".")
    assert f"{int(int_part):,}.{frac}" in html
    assert r["reinvest"]["text"].replace(">", "&gt;") in html
    for f in r["forecast"]["months"]:
        assert f["month"] in html
    assert "2025-12" in html  # история по месяцам
