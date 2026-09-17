#!/usr/bin/env node
/**
 * Resolve and download a wheel (or sdist) from PyPI by package name.
 * Uses Node's OpenSSL TLS stack, which works on this machine.
 *
 * Usage:
 *   node pywheel.js <package> [version] [--out <dir>] [--prefer-win]
 */
const https = require('https');
const fs = require('fs');
const path = require('path');

function get(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 10) return reject(new Error('too many redirects'));
    https
      .get(url, { headers: { 'User-Agent': 'dublador-bootstrap/1.0', Accept: 'application/json,*/*' } }, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
          res.resume();
          return resolve(get(res.headers.location, redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} ${url}`));
        }
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ buffer: Buffer.concat(chunks), headers: res.headers }));
      })
      .on('error', reject);
  });
}

function scoreWheel(name) {
  // Prefer universal wheels, then cp311 win_amd64, then cp311 any, then anything.
  if (/py3-none-any\.whl$/.test(name)) return 100;
  if (/py2\.py3-none-any\.whl$/.test(name)) return 99;
  if (/cp311-cp311-win_amd64\.whl$/.test(name)) return 90;
  if (/cp311-abi3-win_amd64\.whl$/.test(name)) return 88;
  if (/cp311-none-win_amd64\.whl$/.test(name)) return 86;
  if (/cp311.*win_amd64\.whl$/.test(name)) return 80;
  if (/cp311.*\.whl$/.test(name)) return 60;
  if (/py3-none-any/.test(name)) return 50;
  if (/\.whl$/.test(name)) return 10;
  return 0;
}

async function main() {
  const argv = process.argv.slice(2);
  let outDir = 'tools/dist';
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') outDir = argv[++i];
    else rest.push(argv[i]);
  }
  const pkg = rest[0];
  const wantedVersion = rest[1];
  if (!pkg) {
    console.error('usage: node pywheel.js <package> [version] [--out dir]');
    process.exit(2);
  }

  const metaUrl = wantedVersion
    ? `https://pypi.org/pypi/${pkg}/${wantedVersion}/json`
    : `https://pypi.org/pypi/${pkg}/json`;
  const { buffer } = await get(metaUrl);
  const meta = JSON.parse(buffer.toString('utf8'));
  const version = meta.info.version;
  const files = meta.urls || [];
  const wheels = files.filter((f) => f.filename.endsWith('.whl') && !f.yanked);
  if (!wheels.length) {
    console.error(`no wheels for ${pkg} ${version}`);
    process.exit(1);
  }
  wheels.sort((a, b) => scoreWheel(b.filename) - scoreWheel(a.filename));
  const chosen = wheels[0];
  console.log(`${pkg} ${version} -> ${chosen.filename} (${(chosen.size / 1048576).toFixed(2)} MB)`);

  fs.mkdirSync(outDir, { recursive: true });
  const dest = path.join(outDir, chosen.filename);
  if (fs.existsSync(dest) && fs.statSync(dest).size === chosen.size) {
    console.log('already downloaded');
  } else {
    const r = await get(chosen.url);
    fs.writeFileSync(dest, r.buffer);
    console.log('saved', dest);
  }
  console.log('RESOLVED_VERSION=' + version);
  console.log('WHEEL_PATH=' + path.resolve(dest));
}

main().catch((e) => {
  console.error('ERROR: ' + e.message);
  process.exit(1);
});
