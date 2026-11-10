// emotional-memory.js — шкалы trust / comfort / attachment (0..100) для персонализации общения.
const { updateMemory } = require('./fan-memory');

const KEYS = ['trust', 'comfort', 'attachment'];
const MAX_STEP = 20; // одно событие не может сдвинуть шкалу сильнее

function clamp(v) {
  return Math.max(0, Math.min(100, v));
}

function _delta(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(-MAX_STEP, Math.min(MAX_STEP, n));
}

function updateEmotional(device, userId, delta = {}) {
  if (typeof userId !== 'string' || !userId || userId === '__proto__') {
    throw new Error(`некорректный userId: ${JSON.stringify(userId)}`);
  }
  return updateMemory(device, (mem) => {
    const fan = Object.prototype.hasOwnProperty.call(mem.fans, userId) ? mem.fans[userId] : { user_id: userId };
    for (const k of KEYS) {
      fan[k] = clamp((Number(fan[k]) || 0) + _delta(delta[k]));
    }
    fan.emotional_updated_at = Date.now();
    mem.fans[userId] = fan;
    return fan;
  });
}

module.exports = { updateEmotional, clamp };
