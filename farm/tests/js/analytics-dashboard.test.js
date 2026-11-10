const fs = require('fs');
const path = require('path');
const { freshRoot, load, writeJSON } = require('./helpers');

const PNL = {
  month: '2026-01',
  accounts: [{ account_id: 'aria_fight_tiktok', cpa_by_offer: { 1: '30.00', promo: '3.33' }, cpa_gross: '33.33',
    cpa_commission: '5.00', software_sales: 1, software_gross: '19.99', software_fees: '0.88', gross: '53.32',
    net: '47.44' }],
  totals: { cpa_gross: '33.33', cpa_commission: '5.00', software_gross: '19.99', software_fees: '0.88',
    gross: '53.32', net: '47.44' },
  expenses: { gcp: { amount: '10.10', source: 'budget_guard' }, other: { phones: '5.55' }, total: '15.65' },
  pnl: { gross: '53.32', direct_costs: '5.88', expenses: '15.65', net: '-1234.50' },
  summary: { total_accounts: 30, active_accounts: 1, revenue_per_account: '1.78', arpu: '53.32', net_per_account: '1.06' },
  forecast: { method: 'линейная', history_months: ['2026-01'], months: [{ month: '2026-02', net: '5000.00' }] },
  reinvest: { text: 'можно запустить 125 аккаунтов, но потолок 8/мес -> запускаем 8', net_available: '5000.00',
    account_cost: '40.00', possible: 125, cap: 8, launch: 8, launch_cost: '320.00', left_after_launch: '4680.00' },
  alerts: ['<script>alert(1)</script>'],
  data_quality: { pending_cpa_usd: '1.00', refunds_usd: '0.00', skipped_records: 0, duplicates: 0 },
};

let root;
let dash;
beforeEach(() => {
  root = freshRoot();
  jest.spyOn(console, 'log').mockImplementation(() => {});
  dash = load('analytics-dashboard.js');
});
afterEach(() => jest.restoreAllMocks());

test('usd: разряды, минус, без перевода в число', () => {
  expect(dash.usd('1234567.05')).toBe('$1,234,567.05');
  expect(dash.usd('-1234.50')).toBe('−$1,234.50');
  expect(dash.usd('0.30')).toBe('$0.30');
  expect(dash.usd('<b>')).toBe('&lt;b&gt;');
});

test('renderPnl: цифры как в JSON, HTML экранируется', () => {
  const html = dash.renderPnl(PNL, [{ month: '2025-12', gross: '1.00', expenses: '0.00', net: '1.00' }]);
  expect(html).toContain('−$1,234.50');
  expect(html).toContain('$5,000.00');
  expect(html).toContain('запускаем 8');
  expect(html).toContain('promo: $3.33');
  expect(html).toContain('&lt;script&gt;');
  expect(html).not.toContain('<script>');
  expect(html).toContain('2025-12');
});

test('buildDashboard: последний месяц по умолчанию, файл и json, атомарно', async () => {
  const dir = path.join(root, 'data', 'analytics');
  writeJSON(path.join(dir, 'pnl_2025-12.json'), { ...PNL, month: '2025-12', pnl: { ...PNL.pnl, net: '1.00' } });
  writeJSON(path.join(dir, 'pnl_2026-01.json'), PNL);
  writeJSON(path.join(dir, 'pnl_bad-name.json'), {});
  const out = await dash.buildDashboard({ project: 'T' }, { dataDir: path.join(root, 'data'), includeFans: false });
  expect(out).toBe(path.join(dir, 'dashboard.html'));
  const html = fs.readFileSync(out, 'utf8');
  expect(html).toContain('Экономика 2026-01');
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'dashboard.json'), 'utf8'));
  expect(meta.month).toBe('2026-01');
  expect(fs.readdirSync(dir).filter((n) => n.includes('.tmp.'))).toEqual([]);
  expect(await dash.listPnlMonths(dir)).toEqual(['2025-12', '2026-01']);
});

test('buildDashboard без данных — подсказка, не падение', async () => {
  const out = await dash.buildDashboard({}, { dataDir: path.join(root, 'data'), includeFans: false });
  expect(fs.readFileSync(out, 'utf8')).toContain('revenue_tracker.py');
});

test('loadPnl проверяет месяц', async () => {
  await expect(dash.loadPnl(root, '../../etc')).rejects.toThrow(/YYYY-MM/);
});

test('секция фанатов из fan-memory', async () => {
  writeJSON(path.join(root, 'models', 'aria_fight', 'memory.json'), { fans: {
    u1: { lifetime_spend: 20, last_message_at: Date.now(), ppv_sent: [{ status: 'sent', purchased: true, amount: 20 }] } } });
  const rows = dash.collect();
  expect(rows[0]).toMatchObject({ device: 'aria_fight', fans: 1, paying: 1, ppvBought: 1, ppvRevenue: 20 });
  const out = await dash.buildDashboard({}, { dataDir: path.join(root, 'data') });
  expect(fs.readFileSync(out, 'utf8')).toContain('Фанаты (fan-memory)');
});
