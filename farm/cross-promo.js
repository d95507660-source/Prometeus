// cross-promo.js — ежедневный план взаимного продвижения между аккаунтами.
// Каждое устройство упоминает одно другое (по возможности из другой ниши); пары меняются каждый день.
const path = require('path');
const { ROOT, DATA_DIR, readJSON, writeJSONAtomic, stableHash, today } = require('./common');

function _activeDevices(config) {
  const active = new Set(config.mimic?.active_niches || []);
  const devices = readJSON(path.join(ROOT, 'devices.json'), { devices: [] }).devices || [];
  return devices.filter((d) => d.enabled !== false && (!active.size || active.has(d.niche)));
}

function runCrossPromo(config) {
  const day = today(config.timezone || 'UTC');
  const devices = _activeDevices(config)
    .map((d) => ({ d, k: stableHash(`${day}:${d.device_id}`) }))
    .sort((a, b) => a.k - b.k)
    .map((x) => x.d);
  if (devices.length < 2) {
    console.log('[cross-promo] меньше двух устройств — пропуск');
    return [];
  }
  const pairs = [];
  const used = new Set();
  for (let i = 0; i < devices.length; i++) {
    const from = devices[i];
    // сначала ищем партнёра из другой ниши, которого ещё никто не продвигает
    let to = null;
    for (let j = 1; j < devices.length && !to; j++) {
      const c = devices[(i + j) % devices.length];
      if (c.niche !== from.niche && !used.has(c.device_id)) to = c;
    }
    for (let j = 1; j < devices.length && !to; j++) {
      const c = devices[(i + j) % devices.length];
      if (!used.has(c.device_id)) to = c;
    }
    if (!to) to = devices[(i + 1) % devices.length];
    used.add(to.device_id);
    const handles = {};
    for (const [platform, acc] of Object.entries(to.accounts || {})) {
      if (acc.enabled !== false && acc.username) handles[platform] = `@${acc.username}`;
    }
    pairs.push({ from: from.device_id, to: to.device_id, from_niche: from.niche, to_niche: to.niche,
      handles, suggestion: `say hi to my friend ${Object.values(handles)[0] || to.model_name} 💫` });
  }
  const out = path.join(DATA_DIR, 'cross_promo', `${day}.json`);
  writeJSONAtomic(out, { day, pairs });
  console.log(`[cross-promo] ${pairs.length} пар -> ${out}`);
  return pairs;
}

module.exports = { runCrossPromo };
