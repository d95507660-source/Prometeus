// helpers.js — изолированный корень проекта для каждого теста (PROMETHEUS_ROOT во временной папке).
const fs = require('fs');
const os = require('os');
const path = require('path');

function freshRoot() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'prometheus-js-'));
  process.env.PROMETHEUS_ROOT = root;
  jest.resetModules(); // common.js читает PROMETHEUS_ROOT при загрузке
  return root;
}

function load(name) {
  return require(path.join(__dirname, '..', '..', name));
}

function writeJSON(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(data));
}

module.exports = { freshRoot, load, writeJSON };
