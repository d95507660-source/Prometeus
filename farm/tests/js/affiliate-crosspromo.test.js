const fs = require('fs');
const path = require('path');
const { freshRoot, load, writeJSON } = require('./helpers');

let root;
beforeEach(() => {
  root = freshRoot();
  jest.spyOn(console, 'log').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => jest.restoreAllMocks());

describe('affiliate-tracker', () => {
  test('buildLink: только https, slug экранируется', () => {
    const { buildLink } = load('affiliate-tracker.js');
    expect(buildLink('https://p.example/o?ref={model}', 'aria x&y')).toBe('https://p.example/o?ref=aria%20x%26y');
    expect(() => buildLink('http://p.example/{model}', 'a')).toThrow(/https/);
    expect(() => buildLink('javascript:alert(1)//{model}', 'a')).toThrow();
    expect(() => buildLink('not a url {model}', 'a')).toThrow();
  });

  test('trackAffiliate: ссылки по моделям, ошибки по кривым партнёрам, журнал изменений', () => {
    writeJSON(path.join(root, 'data', 'affiliate', 'partners.json'), { partners: [
      { id: 'p1', name: 'Good', enabled: true, cpa: 3, url: 'https://good.example/?m={model}' },
      { id: 'p2', name: 'Bad', enabled: true, cpa: 1, url: 'http://bad.example/{model}' },
      { id: 'p3', name: 'Off', enabled: false, cpa: 9, url: 'https://off.example/{model}' }] });
    writeJSON(path.join(root, 'models', 'aria_fight', 'metadata.json'), { model_name: 'Aria' });
    fs.mkdirSync(path.join(root, 'models', 'mia_fight'), { recursive: true });
    const res = load('affiliate-tracker.js').trackAffiliate({});
    expect(res.partners).toBe(2);
    expect(res.links.aria_fight.p1.url).toBe('https://good.example/?m=aria');
    expect(res.links.mia_fight.p1.url).toBe('https://good.example/?m=mia_fight');
    expect(res.errors).toHaveLength(2);
    const seen = JSON.parse(fs.readFileSync(path.join(root, 'data', 'affiliate', 'seen.json'), 'utf8'));
    expect(Object.keys(seen).sort()).toEqual(['p1', 'p2']);
  });
});

describe('cross-promo', () => {
  const dev = (id, niche) => ({ device_id: id, niche, enabled: true,
    accounts: { tiktok: { username: `${id}.x`, enabled: true } } });

  test('каждое устройство продвигает другое, предпочтительно из другой ниши; каждого продвигают один раз', () => {
    writeJSON(path.join(root, 'devices.json'), { devices: [
      dev('a1', 'fit'), dev('a2', 'fit'), dev('b1', 'yoga'), dev('b2', 'yoga'), dev('c1', 'dance')] });
    const pairs = load('cross-promo.js').runCrossPromo({ timezone: 'UTC', mimic: { active_niches: ['fit', 'yoga', 'dance'] } });
    expect(pairs).toHaveLength(5);
    expect(pairs.every((p) => p.from !== p.to)).toBe(true);
    expect(new Set(pairs.map((p) => p.to)).size).toBe(5);
    expect(pairs.filter((p) => p.from_niche === p.to_niche).length).toBeLessThanOrEqual(1);
    expect(pairs[0].suggestion).toMatch(/@/);
  });

  test('меньше двух устройств — пропуск', () => {
    writeJSON(path.join(root, 'devices.json'), { devices: [dev('a1', 'fit')] });
    expect(load('cross-promo.js').runCrossPromo({})).toEqual([]);
  });
});
