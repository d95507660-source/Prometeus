const fs = require('fs');
const path = require('path');
const { freshRoot, load } = require('./helpers');

let root;
let common;
beforeEach(() => {
  root = freshRoot();
  common = load('common.js');
});

test('PROMETHEUS_ROOT задаёт корень данных', () => {
  expect(common.ROOT).toBe(path.resolve(root));
  expect(common.DATA_DIR).toBe(path.join(root, 'data'));
});

test('writeJSONAtomic пишет файл и не оставляет tmp', () => {
  const f = path.join(common.DATA_DIR, 'x', 'a.json');
  common.writeJSONAtomic(f, { a: 1 });
  expect(JSON.parse(fs.readFileSync(f, 'utf8'))).toEqual({ a: 1 });
  expect(fs.readdirSync(path.dirname(f))).toEqual(['a.json']);
});

test('readJSON: нет файла — fallback, битый — fallback и сообщение', () => {
  const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
  expect(common.readJSON(path.join(root, 'none.json'), { d: 1 })).toEqual({ d: 1 });
  expect(spy).not.toHaveBeenCalled();
  const bad = path.join(root, 'bad.json');
  fs.writeFileSync(bad, '{oops');
  expect(common.readJSON(bad, [])).toEqual([]);
  expect(spy).toHaveBeenCalled();
  spy.mockRestore();
});

test.each(['../etc', 'a b', '', 'x'.repeat(65), 42, null])('assertId отклоняет %p', (v) => {
  expect(() => common.assertId(v)).toThrow(/некорректный/);
});

test('listModelDirs: только безопасные имена, без _скрытых и файлов', () => {
  for (const d of ['aria_fight', '_tmp', 'bad name']) fs.mkdirSync(path.join(common.MODELS_DIR, d), { recursive: true });
  fs.writeFileSync(path.join(common.MODELS_DIR, 'file.txt'), '');
  expect(common.listModelDirs()).toEqual(['aria_fight']);
});

test('today — YYYY-MM-DD, stableHash — детерминирован', () => {
  expect(common.today('UTC')).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  expect(common.stableHash('aria')).toBe(common.stableHash('aria'));
  expect(common.stableHash('aria')).not.toBe(common.stableHash('mia'));
});

test('queueOutbox дописывает строки JSONL', () => {
  const f = common.queueOutbox('ppv', { user: 'u1' });
  common.queueOutbox('ppv', { user: 'u2' });
  const lines = fs.readFileSync(f, 'utf8').trim().split('\n').map(JSON.parse);
  expect(lines.map((l) => l.user)).toEqual(['u1', 'u2']);
  expect(lines[0].kind).toBe('ppv');
});
