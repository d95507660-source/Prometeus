const fs = require('fs');
const path = require('path');
const { freshRoot, load } = require('./helpers');

let root;
let fm;
let em;
beforeEach(() => {
  root = freshRoot();
  fm = load('fan-memory.js');
  em = load('emotional-memory.js');
});

const memFile = () => path.join(root, 'models', 'aria_fight', 'memory.json');

test('upsert, recordMessage, getFan', () => {
  fm.upsertFan('aria_fight', 'u1', { name_hint: 'x' });
  fm.recordMessage('aria_fight', 'u1');
  const fan = fm.recordMessage('aria_fight', 'u1');
  expect(fan.messages_count).toBe(2);
  expect(fm.getFan('aria_fight', 'u1').name_hint).toBe('x');
  expect(fm.getFan('aria_fight', 'nobody')).toBeNull();
  expect(fs.existsSync(`${memFile()}.lock`)).toBe(false); // блокировка снята
});

test('recordPurchase: сумма с округлением, отмечает последний платный PPV', () => {
  fm.upsertFan('aria_fight', 'u1', { ppv_sent: [
    { ppv_id: 'a', amount: 10, status: 'sent' }, { ppv_id: 'teaser', amount: 0, status: 'sent' }] });
  fm.recordPurchase('aria_fight', 'u1', { amount: 0.1 });
  const fan = fm.recordPurchase('aria_fight', 'u1', { amount: 0.2 });
  expect(fan.lifetime_spend).toBe(0.3);
  expect(fan.ppv_sent[0].purchased).toBe(true);
  expect(fan.ppv_sent[1].purchased).toBeUndefined(); // бесплатный тизер — не покупка
  expect(() => fm.recordPurchase('aria_fight', 'u1', { amount: -1 })).toThrow(/сумма/);
});

test.each(['__proto__', 'constructor', 'a b', '../x', ''])('userId %p отклоняется', (id) => {
  expect(() => fm.upsertFan('aria_fight', id, {})).toThrow(/userId/);
});

test('device с ../ отклоняется', () => {
  expect(() => fm.loadMemory('../etc')).toThrow(/некорректный/);
});

test('битый memory.json не затирается молча — копия .corrupt-*', () => {
  fs.mkdirSync(path.dirname(memFile()), { recursive: true });
  fs.writeFileSync(memFile(), '{broken');
  const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
  expect(fm.loadMemory('aria_fight')).toEqual({ fans: {} });
  spy.mockRestore();
  expect(fs.readdirSync(path.dirname(memFile())).some((n) => n.startsWith('memory.json.corrupt-'))).toBe(true);
});

test('устаревшая блокировка (упавший процесс) снимается', () => {
  fs.mkdirSync(path.dirname(memFile()), { recursive: true });
  fs.writeFileSync(`${memFile()}.lock`, '99999');
  const old = (Date.now() - 60_000) / 1000;
  fs.utimesSync(`${memFile()}.lock`, old, old);
  expect(fm.recordMessage('aria_fight', 'u1').messages_count).toBe(1);
});

test('emotional-memory: шаг не больше 20, шкала 0..100', () => {
  expect(em.updateEmotional('aria_fight', 'u1', { trust: 500, comfort: -5, attachment: 'x' }))
    .toMatchObject({ trust: 20, comfort: 0, attachment: 0 });
  for (let i = 0; i < 10; i++) em.updateEmotional('aria_fight', 'u1', { trust: 20 });
  expect(fm.getFan('aria_fight', 'u1').trust).toBe(100);
  expect(em.clamp(-3)).toBe(0);
  expect(() => em.updateEmotional('aria_fight', '__proto__', {})).toThrow();
});
