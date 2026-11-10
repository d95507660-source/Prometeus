// affiliate-tracker.js — партнёрские ссылки для каждой модели + журнал изменений партнёров.
const path = require('path');
const { DATA_DIR, listModelDirs, modelMetadata, readJSON, writeJSONAtomic } = require('./common');

function buildLink(template, modelSlug) {
  const url = String(template).replace(/\{model\}/g, encodeURIComponent(modelSlug));
  const parsed = new URL(url); // бросит ошибку на кривой адрес
  if (parsed.protocol !== 'https:') throw new Error(`разрешены только https-ссылки: ${url}`);
  return parsed.toString();
}

function trackAffiliate(config) {
  const dir = path.join(DATA_DIR, 'affiliate');
  const partners = (readJSON(path.join(dir, 'partners.json'), { partners: [] }).partners || [])
    .filter((p) => p && p.enabled && p.id && p.url);
  const seen = readJSON(path.join(dir, 'seen.json'), {});
  const links = {};
  const errors = [];

  for (const p of partners) {
    const sig = `${p.url}|${p.cpa}`;
    if (!seen[p.id] || seen[p.id].sig !== sig) {
      seen[p.id] = { sig, first_seen: seen[p.id]?.first_seen || Date.now(), updated: Date.now() };
      console.log(`[affiliate] партнёр ${p.name} (${p.id}) cpa=${p.cpa} — новый или изменён`);
    }
    for (const device of listModelDirs()) {
      const slug = (modelMetadata(device).model_name || device).toLowerCase();
      try {
        (links[device] ||= {})[p.id] = { name: p.name, cpa: Number(p.cpa) || 0, url: buildLink(p.url, slug) };
      } catch (e) {
        errors.push(`${p.id}/${device}: ${e.message}`);
      }
    }
  }
  writeJSONAtomic(path.join(dir, 'seen.json'), seen);
  writeJSONAtomic(path.join(dir, 'links.json'), { generated_at: Date.now(), links });
  if (errors.length) console.error('[affiliate] ошибки:', errors.join('; '));
  console.log(`[affiliate] партнёров: ${partners.length}, моделей: ${Object.keys(links).length}`);
  return { partners: partners.length, links, errors };
}

module.exports = { trackAffiliate, buildLink };
