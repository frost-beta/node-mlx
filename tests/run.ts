import fs from 'node:fs';
import path from 'node:path';
import Mocha from 'mocha';

const mocha = new Mocha();

for (const f of fs.readdirSync(__dirname)) {
  if (f.endsWith('.spec.ts'))
    mocha.addFile(path.join(__dirname, f));
}
mocha.run(process.exit);
