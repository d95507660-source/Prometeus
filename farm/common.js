// common.js — общие помощники для JS-части: пути от корня проекта, атомарная запись JSON, очередь исходящих.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = __dirname;
const MODELS_DIR = path.join(ROOT, 'models');
const DATA_DIR = path.join(ROOT, 'data');
const SAFE_ID = /^[A-Za-z0-9_\-]{1,64}$/;

function loadConfig(file = path.join(ROOT, 'config.json')) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function readJSON(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e) {
    if (e.code !== 'ENOENT') console.error(`[common] не читается ${file}: ${e.message}`);
    return fallback;
  }
}

function writeJSONAtomic(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp.${process.pid}.${crypto.randomBytes(4).toString('hex')}`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
  fs.renameSync(tmp, file);
}

function today(tz = 'UTC') {
  // en-CA даёт формат YYYY-MM-DD
  return new Intl.DateTimeFormat('en-CA', { timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit' })
    .format(new Date());
}

function assertId(value, what = 'id') {
  if (typeof value !== 'string' || !SAFE_ID.test(value)) {
    throw new Error(`некорректный ${what}: ${JSON.stringify(value)}`);
  }
  return value;
}

function listModelDirs() {
  if (!fs.existsSync(MODELS_DIR)) return [];
  return fs.readdirSync(MODELS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith('_') && SAFE_ID.test(d.name))
    .map((d) => d.name);
}

function modelMetadata(device) {
  return readJSON(path.join(MODELS_DIR, assertId(device, 'device'), 'metadata.json'), {});
}

// Исходящие сообщения/предложения, которые не отправлены автоматически, — на ручную проверку.
function queueOutbox(kind, item, tz = 'UTC') {
  const file = path.join(DATA_DIR, 'outbox', `${today(tz)}.jsonl`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, JSON.stringify({ kind, ts: Date.now(), ...item }) + '\n');
  return file;
}

function stableHash(str) {
  return parseInt(crypto.createHash('sha256').update(String(str)).digest('hex').slice(0, 8), 16);
}

module.exports = {
  ROOT, MODELS_DIR, DATA_DIR, SAFE_ID,
  loadConfig, readJSON, writeJSONAtomic, today, assertId, listModelDirs, modelMetadata,
  queueOutbox, stableHash,
};
