// analytics-dashboard.js — веб-панель экономики проекта (data/analytics/dashboard.html).
//
// Источник цифр — data/analytics/pnl_<YYYY-MM>.json от revenue_tracker.py (Decimal, до цента).
// Панель ничего не пересчитывает: показывает таблицу по аккаунтам, сводку, прогноз и реинвест-план
// ровно так, как их посчитал Python, поэтому расхождений в копейках между JSON и HTML нет.
// Ниже — прежняя сводка по фанатам (collect), если есть data из fan-memory.
//
// Запуск:  node analytics-dashboard.js [--month 2026-01]   (Node 18+)
// buildDashboard теперь async и возвращает Promise<путь> — вызывающий код должен делать await.
const path = require('path');
const fsp = require('fs/promises');
const crypto = require('crypto');
const { DATA_DIR, listModelDirs, writeJSONAtomic } = require('./common');
const { loadMemory } = require('./fan-memory');

const MONTH_RE = /^\d{4}-(0[1-9]|1[0-2])$/;
const MONEY_RE = /^-?\d+\.\d{2}$/;

function esc(v) {
  return String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

// "-1234.50" → "−$1,234.50" без перевода в число (точность строки сохраняется)
function usd(s) {
  const str = String(s ?? '0.00');
  if (!MONEY_RE.test(str)) return esc(str);
  const neg = str.startsWith('-');
  const [int, frac] = (neg ? str.slice(1) : str).split('.');
  const grouped = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return `${neg ? '−' : ''}$${grouped}.${frac}`;
}

async function writeFileAtomic(file, text) {
  await fsp.mkdir(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp.${process.pid}.${crypto.randomBytes(4).toString('hex')}`;
  await fsp.writeFile(tmp, text);
  await fsp.rename(tmp, file);
}

async function listPnlMonths(analyticsDir) {
  let names = [];
  try {
    names = await fsp.readdir(analyticsDir);
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
  return names
    .map((n) => /^pnl_(\d{4}-\d{2})\.json$/.exec(n))
    .filter((m) => m && MONTH_RE.test(m[1]))
    .map((m) => m[1])
    .sort();
}

async function loadPnl(analyticsDir, month) {
  if (!MONTH_RE.test(month)) throw new Error(`месяц должен быть YYYY-MM: ${JSON.stringify(month)}`);
  const text = await fsp.readFile(path.join(analyticsDir, `pnl_${month}.json`), 'utf8');
  return JSON.parse(text);
}

// --- прежняя сводка по фанатам (fan-memory) -------------------------------------
function collect() {
  const now = Date.now();
  const rows = [];
  for (const device of listModelDirs()) {
    const mem = loadMemory(device);
    const r = { device, fans: 0, paying: 0, active7d: 0, spend: 0, ppvSent: 0, ppvBought: 0, ppvRevenue: 0 };
    for (const fan of Object.values(mem.fans)) {
      r.fans += 1;
      r.spend += num(fan.lifetime_spend);
      if (num(fan.lifetime_spend) > 0) r.paying += 1;
      if (now - num(fan.last_message_at) < 7 * 864e5) r.active7d += 1;
      for (const o of fan.ppv_sent || []) {
        if (o.status !== 'sent') continue;
        r.ppvSent += 1;
        if (o.purchased) { r.ppvBought += 1; r.ppvRevenue += num(o.amount); }
      }
    }
    r.conversion = r.ppvSent ? (r.ppvBought / r.ppvSent) * 100 : 0;
    r.arppu = r.paying ? r.spend / r.paying : 0;
    rows.push(r);
  }
  rows.sort((a, b) => b.spend - a.spend);
  return rows;
}

function renderFans(rows) {
  if (!rows.length) return '';
  const m = (v) => `$${num(v).toFixed(2)}`;
  const tr = (r) => `<tr><td>${esc(r.device)}</td><td>${r.fans}</td><td>${r.paying}</td><td>${r.active7d}</td>`
    + `<td>${r.ppvSent}</td><td>${r.ppvBought}</td><td>${r.conversion.toFixed(1)}%</td>`
    + `<td>${m(r.ppvRevenue)}</td><td>${m(r.arppu)}</td><td>${m(r.spend)}</td></tr>`;
  return `<h3>Фанаты (fan-memory)</h3><table><tr><th>Модель</th><th>Фанаты</th><th>Платящие</th>`
    + `<th>Активны 7д</th><th>PPV отпр.</th><th>PPV куплено</th><th>Конверсия</th><th>Выручка PPV</th>`
    + `<th>ARPPU</th><th>Всего</th></tr>${rows.map(tr).join('\n')}</table>`;
}

// --- P&L --------------------------------------------------------------------------
function renderPnl(pnl, history = []) {
  const offers = (a) => Object.entries(a.cpa_by_offer || {})
    .map(([o, v]) => `${esc(o)}: ${usd(v)}`).join('<br>') || '—';
  const rows = (pnl.accounts || []).map((a) => `<tr><td>${esc(a.account_id)}</td><td>${offers(a)}</td>`
    + `<td>${usd(a.cpa_gross)}</td><td>${a.software_sales}</td><td>${usd(a.software_gross)}</td>`
    + `<td>${usd(a.gross)}</td><td>${usd(a.cpa_commission)}</td><td>${usd(a.software_fees)}</td>`
    + `<td class="${String(a.net).startsWith('-') ? 'neg' : ''}">${usd(a.net)}</td></tr>`).join('\n');
  const t = pnl.totals || {};
  const s = pnl.summary || {};
  const p = pnl.pnl || {};
  const ex = pnl.expenses || { gcp: {}, other: {} };
  const other = Object.entries(ex.other || {})
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${usd(v)}</td></tr>`).join('');
  const fc = (pnl.forecast && pnl.forecast.months) || [];
  const r = pnl.reinvest || {};
  const dq = pnl.data_quality || {};
  const hist = history.map((h) => `<tr><td>${esc(h.month)}</td><td>${usd(h.gross)}</td>`
    + `<td>${usd(h.expenses)}</td><td>${usd(h.net)}</td></tr>`).join('');
  const alerts = (pnl.alerts || []).map((a) => `<li>${esc(a)}</li>`).join('');

  return `<h2>Экономика ${esc(pnl.month)}</h2>
${alerts ? `<ul class="alerts">${alerts}</ul>` : ''}
<div class="cards">
  <div><b>${usd(p.gross)}</b>валовая</div>
  <div><b>${usd(p.direct_costs)}</b>комиссии и сборы</div>
  <div><b>${usd(p.expenses)}</b>расходы</div>
  <div><b class="${String(p.net).startsWith('-') ? 'neg' : ''}">${usd(p.net)}</b>чистыми</div>
  <div><b>${esc(s.total_accounts)}</b>аккаунтов (с доходом ${esc(s.active_accounts)})</div>
  <div><b>${usd(s.revenue_per_account)}</b>доход на аккаунт</div>
  <div><b>${usd(s.arpu)}</b>ARPU (на аккаунт с доходом)</div>
</div>
<h3>По аккаунтам</h3>
<table><tr><th>Аккаунт</th><th>CPA по офферам</th><th>CPA</th><th>Продаж софта</th><th>Софт</th>
<th>Валовая</th><th>Комиссия CPA</th><th>Сборы за продажи</th><th>Чистыми</th></tr>
${rows}
<tr class="total"><td>ИТОГО</td><td></td><td>${usd(t.cpa_gross)}</td><td></td><td>${usd(t.software_gross)}</td>
<td>${usd(t.gross)}</td><td>${usd(t.cpa_commission)}</td><td>${usd(t.software_fees)}</td><td>${usd(t.net)}</td></tr>
</table>
<h3>Расходы</h3>
<table><tr><th>Статья</th><th>Сумма</th></tr>
<tr><td>GCP (${esc((ex.gcp || {}).source || '')})</td><td>${usd((ex.gcp || {}).amount)}</td></tr>${other}
<tr class="total"><td>Всего</td><td>${usd(ex.total)}</td></tr></table>
<h3>Прогноз (${esc((pnl.forecast || {}).method || '')}, по ${((pnl.forecast || {}).history_months || []).length} мес.)</h3>
<table><tr><th>Месяц</th><th>Чистыми</th></tr>
${fc.map((f) => `<tr><td>${esc(f.month)}</td><td>${usd(f.net)}</td></tr>`).join('')}</table>
<h3>Реинвест-план</h3>
<p>${esc(r.text || '')}</p>
<table><tr><th>Доступно</th><th>Цена аккаунта</th><th>Можно</th><th>Потолок</th><th>Запускаем</th><th>Стоимость</th><th>Остаток</th></tr>
<tr><td>${usd(r.net_available)}</td><td>${usd(r.account_cost)}</td><td>${esc(r.possible)}</td><td>${esc(r.cap)}</td>
<td>${esc(r.launch)}</td><td>${usd(r.launch_cost)}</td><td>${usd(r.left_after_launch)}</td></tr></table>
${hist ? `<h3>По месяцам</h3><table><tr><th>Месяц</th><th>Валовая</th><th>Расходы</th><th>Чистыми</th></tr>${hist}</table>` : ''}
<p class="muted">CPA в ожидании: ${usd(dq.pending_cpa_usd)} · возвраты: ${usd(dq.refunds_usd)} · пропущено записей: ${esc(dq.skipped_records ?? 0)} · дублей: ${esc(dq.duplicates ?? 0)}</p>`;
}

async function buildDashboard(config = {}, { month = null, dataDir = DATA_DIR, includeFans = true } = {}) {
  const analyticsDir = path.join(dataDir, 'analytics');
  const months = await listPnlMonths(analyticsDir);
  const target = month || months[months.length - 1];
  let body = '<p>Нет данных P&amp;L: запусти <code>python revenue_tracker.py</code>.</p>';
  let pnl = null;
  if (target) {
    pnl = await loadPnl(analyticsDir, target);
    const history = [];
    for (const m of months) {
      const h = m === target ? pnl : await loadPnl(analyticsDir, m);
      history.push({ month: m, ...h.pnl });
    }
    body = renderPnl(pnl, history);
  }
  const fans = includeFans ? collect() : [];
  const title = esc(config.project || 'Prometheus');
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>${title} — экономика</title>
<style>body{font-family:monospace;background:#111;color:#eee;padding:20px;max-width:1200px}
table{border-collapse:collapse;margin-bottom:16px}td,th{padding:6px 12px;border:1px solid #333;text-align:right;vertical-align:top}
td:first-child,th:first-child{text-align:left}th{background:#222}tr.total{font-weight:bold;background:#1a1a1a}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.cards div{background:#1a1a1a;padding:10px 14px;min-width:150px}
.cards b{display:block;font-size:1.3em}.neg{color:#ff6b6b}.alerts{color:#ffcc66}.muted{color:#888}</style></head><body>
<h1>${title}</h1><p class="muted">Обновлено: ${esc(new Date().toISOString())}</p>
${body}
${renderFans(fans)}
</body></html>`;

  const out = path.join(analyticsDir, 'dashboard.html');
  await writeFileAtomic(out, html);
  writeJSONAtomic(path.join(analyticsDir, 'dashboard.json'), { ts: Date.now(), month: target || null, pnl, fans });
  console.log('[analytics] dashboard ->', out);
  return out;
}

module.exports = { buildDashboard, collect, renderPnl, loadPnl, listPnlMonths, usd };

if (require.main === module) {
  (async () => {
    const i = process.argv.indexOf('--month');
    const month = i > 0 ? process.argv[i + 1] : null;
    if (month && !MONTH_RE.test(month)) throw new Error('--month должен быть YYYY-MM');
    let config = {};
    try {
      config = JSON.parse(await fsp.readFile(path.join(__dirname, 'config.json'), 'utf8'));
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
    }
    await buildDashboard(config, { month });
  })().catch((e) => {
    console.error('[analytics] ошибка:', e.message);
    process.exit(1);
  });
}
