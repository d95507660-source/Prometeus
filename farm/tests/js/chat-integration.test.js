// Сеть не используется: global.fetch подменён, таймеры — фейковые (повторы ждут 1–2 с).
const { load } = require('./helpers');

function response(status, body, headers = {}) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (k) => headers[k.toLowerCase()] ?? null },
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

let chat;
beforeEach(() => {
  jest.resetModules();
  process.env.FANVUE_TOKEN = 'tok';
  process.env.FANVUE_BASE = 'https://fanvue.test/v1/';
  process.env.ANLORA_BASE_URL = 'https://anlora.test';
  global.fetch = jest.fn();
  chat = load('chat-integration.js');
});
afterEach(() => {
  jest.useRealTimers();
  delete global.fetch;
});

test('успешная отправка: адрес, токен, тело', async () => {
  global.fetch.mockResolvedValue(response(200, { ok: 1 }, { 'content-type': 'application/json' }));
  await expect(chat.sendFanvueReply('aria_fight', 'u1', 'hi')).resolves.toEqual({ ok: 1 });
  const [url, opts] = global.fetch.mock.calls[0];
  expect(url).toBe('https://fanvue.test/v1/models/aria_fight/messages');
  expect(opts.headers.Authorization).toBe('Bearer tok');
  expect(JSON.parse(opts.body)).toEqual({ user_id: 'u1', text: 'hi' });
});

test('429 → повтор с паузой → успех', async () => {
  jest.useFakeTimers();
  global.fetch
    .mockResolvedValueOnce(response(429, {}, { 'retry-after': '1' }))
    .mockResolvedValueOnce(response(200, { ok: 2 }, { 'content-type': 'application/json' }));
  const p = chat.sendFanvueReply('aria_fight', 'u1', 'hi');
  await jest.advanceTimersByTimeAsync(1000);
  await expect(p).resolves.toEqual({ ok: 2 });
  expect(global.fetch).toHaveBeenCalledTimes(2);
});

test('4xx (кроме 429) не повторяется', async () => {
  global.fetch.mockResolvedValue(response(403, { error: 'forbidden' }));
  await expect(chat.sendFanvueReply('aria_fight', 'u1', 'hi')).rejects.toThrow(/HTTP 403/);
  expect(global.fetch).toHaveBeenCalledTimes(1);
});

test('PPV: платёж повторяется максимум один раз, цена проверяется', async () => {
  jest.useFakeTimers();
  global.fetch.mockResolvedValue(response(500, {}));
  const p = chat.sendPPV('aria_fight', 'u1', 9.999, 'https://cdn/x.mp4');
  const assertion = expect(p).rejects.toThrow(/HTTP 500/);
  await jest.advanceTimersByTimeAsync(5000);
  await assertion;
  expect(global.fetch).toHaveBeenCalledTimes(2);
  await expect(chat.sendPPV('aria_fight', 'u1', -5, 'x')).rejects.toThrow(/цена/);
  await expect(chat.sendPPV('aria_fight', 'u1', 'abc', 'x')).rejects.toThrow(/цена/);
});

test('проверка id, пустого текста и токена', async () => {
  await expect(chat.sendFanvueReply('../admin', 'u1', 'hi')).rejects.toThrow(/modelId/);
  await expect(chat.sendFanvueReply('aria_fight', 'u1', '   ')).rejects.toThrow(/пустое/);
  delete process.env.FANVUE_TOKEN;
  await expect(chat.sendFanvueReply('aria_fight', 'u1', 'hi')).rejects.toThrow(/FANVUE_TOKEN/);
  expect(global.fetch).not.toHaveBeenCalled();
});

test('Anlora: без ANLORA_BASE_URL — понятная ошибка', async () => {
  delete process.env.ANLORA_BASE_URL;
  jest.resetModules();
  const c = load('chat-integration.js');
  await expect(c.sendAnloraChat('aria_fight', 'u1', 'hi')).rejects.toThrow(/ANLORA_BASE_URL/);
});
