#!/usr/bin/env node
/**
 * HTTPS downloader that bypasses the broken .NET/Schannel TLS stack on this
 * machine (PowerShell's Invoke-WebRequest fails; Node's OpenSSL works).
 *
 * Usage:
 *   node download.js <url> <outfile> [--sha256 <hex>]
 */
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { URL } = require('url');

const MAX_REDIRECTS = 10;

function parseArgs(argv) {
  const args = { url: null, out: null, sha256: null };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--sha256') args.sha256 = argv[++i];
    else rest.push(argv[i]);
  }
  args.url = rest[0];
  args.out = rest[1];
  return args;
}

function fetchToFile(url, outPath, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > MAX_REDIRECTS) return reject(new Error('too many redirects'));
    const u = new URL(url);
    const mod = u.protocol === 'http:' ? http : https;
    const req = mod.get(
      url,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) dublador-bootstrap/1.0',
          Accept: '*/*',
        },
      },
      (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
          res.resume();
          const next = new URL(res.headers.location, url).toString();
          return resolve(fetchToFile(next, outPath, redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        const total = parseInt(res.headers['content-length'] || '0', 10);
        let received = 0;
        let lastPct = -1;
        const tmp = outPath + '.part';
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        const ws = fs.createWriteStream(tmp);
        res.on('data', (chunk) => {
          received += chunk.length;
          if (total) {
            const pct = Math.floor((received / total) * 100);
            if (pct >= lastPct + 10) {
              lastPct = pct;
              process.stderr.write(`  ...${pct}% (${(received / 1048576).toFixed(1)} MB)\n`);
            }
          }
        });
        res.pipe(ws);
        ws.on('finish', () => {
          ws.close(() => {
            fs.renameSync(tmp, outPath);
            resolve({ bytes: received });
          });
        });
        ws.on('error', reject);
      }
    );
    req.on('error', reject);
    req.setTimeout(120000, () => req.destroy(new Error('timeout')));
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.url || !args.out) {
    console.error('usage: node download.js <url> <outfile> [--sha256 <hex>]');
    process.exit(2);
  }
  if (fs.existsSync(args.out) && fs.statSync(args.out).size > 0) {
    console.log(`already exists: ${args.out}`);
  } else {
    console.log(`downloading ${args.url}`);
    const { bytes } = await fetchToFile(args.url, args.out);
    console.log(`saved ${args.out} (${(bytes / 1048576).toFixed(1)} MB)`);
  }
  if (args.sha256) {
    const h = crypto.createHash('sha256');
    h.update(fs.readFileSync(args.out));
    const got = h.digest('hex');
    if (got.toLowerCase() !== args.sha256.toLowerCase()) {
      console.error(`SHA256 MISMATCH\n  expected ${args.sha256}\n  got      ${got}`);
      process.exit(1);
    }
    console.log('sha256 OK');
  }
}

main().catch((e) => {
  console.error('ERROR: ' + e.message);
  process.exit(1);
});
