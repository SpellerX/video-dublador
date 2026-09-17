const os = require('os');
const fs = require('fs');
console.log('=== SYSTEM PROBE ===');
console.log('CPU       :', os.cpus()[0].model);
console.log('Cores     :', os.cpus().length, '(logical)');
console.log('Total RAM :', (os.totalmem() / 1073741824).toFixed(1), 'GB');
console.log('Free RAM  :', (os.freemem() / 1073741824).toFixed(1), 'GB');
console.log('Arch      :', os.arch(), os.platform(), os.release());
for (const d of ['C:', 'N:']) {
  try {
    const s = fs.statfsSync(d + '\\');
    console.log(`Disk ${d}   : ${(s.bsize * s.bavail / 1073741824).toFixed(1)} GB free of ${(s.bsize * s.blocks / 1073741824).toFixed(1)} GB`);
  } catch (e) {
    console.log(`Disk ${d}   : unavailable (${e.code || e.message})`);
  }
}
