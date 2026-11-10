// chat-integration.js — Fanvue (сообщения, PPV) и Anlora (генерация ответа в чате).
// Нужен Node.js >= 18 (встроенный fetch).
// Адреса API задаются через .env: проверь их по документации платформ перед включением auto_send.
const FANVUE_BASE = (process.env.FANVUE_BASE || 'https://api.fanvue.com/v1').replace(/\/+$/, '');
const ANLORA_BASE = (process.env.ANLORA_BASE_URL || '').replace(/\/+$/, '');
const TIMEOUT_MS = Number(process.env.HTTP_TIMEOUT_MS || 20000);

function _sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function _request(url, opts, { retries = 3 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetch(url, { ...opts, signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (r.status === 429 || r.status >= 500) {
        const retryAfter = Number(r.headers.get('retry-after')) || 2 ** attempt;
        lastErr = new Error(`${url}: HTTP ${r.status}`);
        if (attempt < retries) { await _sleep(Math.min(retryAfter, 30) * 1000); continue; }
        throw lastErr;
      }
      if (!r.ok) {
        const body = (await r.text()).slice(0, 300);
        throw new Error(`${url}: HTTP ${r.status} ${body}`);
      }
      const type = r.headers.get('content-type') || '';
      return type.includes('application/json') ? r.json() : r.text();
    } catch (e) {
      lastErr = e;
      const retriable = e.name === 'TimeoutError' || e.name === 'AbortError' || e.code === 'ECONNRESET'
        || /HTTP (429|5\d\d)/.test(e.message);
      if (!retriable || attempt >= retries) throw e;
      await _sleep(1000 * 2 ** attempt);
    }
  }
  throw lastErr;
}

function _fanvueToken() {
  const t = process.env.FANVUE_TOKEN;
  if (!t) throw new Error('FANVUE_TOKEN не задан');
  return t;
}

function _id(v, what) {
  if (typeof v !== 'string' || !/^[A-Za-z0-9_\-:.@]{1,128}$/.test(v)) throw new Error(`некорректный ${what}`);
  return encodeURIComponent(v);
}

async function sendFanvueReply(modelId, userId, text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('пустое сообщение');
  return _request(`${FANVUE_BASE}/models/${_id(modelId, 'modelId')}/messages`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${_fanvueToken()}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, text: text.slice(0, 4000) }),
  });
}

async function sendPPV(modelId, userId, price, mediaUrl, caption = '') {
  const p = Number(price);
  if (!Number.isFinite(p) || p <= 0) throw new Error(`некорректная цена: ${price}`);
  return _request(`${FANVUE_BASE}/models/${_id(modelId, 'modelId')}/ppv`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${_fanvueToken()}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, price: Math.round(p * 100) / 100, media_url: mediaUrl, caption }),
  }, { retries: 1 }); // платёжные действия повторяем осторожно, чтобы не задвоить
}

async function sendAnloraChat(modelId, userId, text, context = {}) {
  if (!ANLORA_BASE) throw new Error('ANLORA_BASE_URL не задан');
  const headers = { 'Content-Type': 'application/json' };
  if (process.env.ANLORA_API_KEY) headers.Authorization = `Bearer ${process.env.ANLORA_API_KEY}`;
  return _request(`${ANLORA_BASE}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model_id: modelId, user_id: userId, message: String(text).slice(0, 4000), context }),
  });
}

module.exports = { sendFanvueReply, sendPPV, sendAnloraChat };
