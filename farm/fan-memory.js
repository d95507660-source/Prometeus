// fan-memory.js — память о фанатах: атомарная запись + блокировка между процессами.
const fs = require('fs');
const path = require('path');
const { MODELS_DIR, assertId, writeJSONAtomic } = require('./common');

const USER_ID = /^[A-Za-z0-9_\-:.@]{1,128}$/;
const LOCK_STALE_MS = 30_000;
const LOCK_WAIT_MS = 10_000;

function _file(device) {
  return path.join(MODELS_DIR, assertId(device, 'device'), 'memory.json');
}

function _assertUser(userId) {
  if (typeof userId !== 'string' || !USER_ID.test(userId) || userId === '__proto__'
      || userId === 'constructor' || userId === 'prototype') {
    throw new Error(`некорректный userId: ${JSON.stringify(userId)}`);
  }
  return userId;
}

function _sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function _withLock(file, fn) {
  const lock = `${file}.lock`;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const deadline = Date.now() + LOCK_WAIT_MS;
  let fd;
  for (;;) {
    try {
      fd = fs.openSync(lock, 'wx');
      fs.writeSync(fd, String(process.pid));
      break;
    } catch (e) {
      if (e.code !== 'EEXIST') throw e;
      try {
        if (Date.now() - fs.statSync(lock).mtimeMs > LOCK_STALE_MS) { fs.unlinkSync(lock); continue; }
      } catch { continue; }
      if (Date.now() > deadline) throw new Error(`не удалось заблокировать ${file}`);
      _sleep(50);
    }
  }
  try {
    return fn();
  } finally {
    fs.closeSync(fd);
    try { fs.unlinkSync(lock); } catch { /* уже удалён */ }
  }
}

function _normalize(mem) {
  if (!mem || typeof mem !== 'object' || Array.isArray(mem)) mem = {};
  if (!mem.fans || typeof mem.fans !== 'object' || Array.isArray(mem.fans)) mem.fans = {};
  return mem;
}

function loadMemory(device) {
  const f = _file(device);
  if (!fs.existsSync(f)) return { fans: {} };
  const text = fs.readFileSync(f, 'utf8');
  try {
    return _normalize(JSON.parse(text));
  } catch (e) {
    // не затираем испорченный файл молча: сохраняем копию для ручного восстановления
    const backup = `${f}.corrupt-${Date.now()}`;
    fs.copyFileSync(f, backup);
    console.error(`[fan-memory] ${f} повреждён (${e.message}), копия: ${backup}`);
    return { fans: {} };
  }
}

function saveMemory(device, mem) {
  writeJSONAtomic(_file(device), _normalize(mem));
}

// Чтение-изменение-запись под блокировкой: без потерянных обновлений между процессами.
function updateMemory(device, mutator) {
  const f = _file(device);
  return _withLock(f, () => {
    const mem = loadMemory(device);
    const result = mutator(mem);
    saveMemory(device, mem);
    return result;
  });
}

function getFan(device, userId) {
  _assertUser(userId);
  const fans = loadMemory(device).fans;
  return Object.prototype.hasOwnProperty.call(fans, userId) ? fans[userId] : null;
}

function upsertFan(device, userId, patch) {
  _assertUser(userId);
  return updateMemory(device, (mem) => {
    const prev = Object.prototype.hasOwnProperty.call(mem.fans, userId) ? mem.fans[userId] : { user_id: userId };
    mem.fans[userId] = { ...prev, ...patch, user_id: userId, updated_at: Date.now() };
    return mem.fans[userId];
  });
}

// Входящее сообщение от фаната: счётчик и время последней активности.
function recordMessage(device, userId) {
  _assertUser(userId);
  return updateMemory(device, (mem) => {
    const fan = mem.fans[userId] || { user_id: userId };
    fan.messages_count = (Number(fan.messages_count) || 0) + 1;
    fan.last_message_at = Date.now();
    mem.fans[userId] = fan;
    return fan;
  });
}

// Покупка (из вебхука платформы): сумма + отметка в последнем подходящем PPV-предложении.
function recordPurchase(device, userId, { amount, ppvId = null }) {
  _assertUser(userId);
  const sum = Number(amount);
  if (!Number.isFinite(sum) || sum < 0) throw new Error(`некорректная сумма: ${amount}`);
  return updateMemory(device, (mem) => {
    const fan = mem.fans[userId] || { user_id: userId };
    fan.lifetime_spend = Math.round(((Number(fan.lifetime_spend) || 0) + sum) * 100) / 100;
    fan.purchases = (fan.purchases || []).concat([{ amount: sum, ppv_id: ppvId, ts: Date.now() }]);
    const sent = fan.ppv_sent || [];
    for (let i = sent.length - 1; i >= 0; i--) {
      const paidOffer = Number(sent[i].amount) > 0; // бесплатный тизер покупкой не считается
      if (paidOffer && !sent[i].purchased && (!ppvId || sent[i].ppv_id === ppvId)) { sent[i].purchased = true; break; }
    }
    mem.fans[userId] = fan;
    return fan;
  });
}

module.exports = { loadMemory, saveMemory, updateMemory, getFan, upsertFan, recordMessage, recordPurchase };
