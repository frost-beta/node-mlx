import fs from 'node:fs';
import path from 'node:path';
import util from 'node:util';
import Mocha from 'mocha';
import tf from '@tensorflow/tfjs';

import {core as mx} from '..';

// Do not warn about using pure-js tensorflow.
tf.ENV.set('IS_TEST', true);

// FIXME(zcbenz): Compilation fails on QEMU in CI.
if (process.env.CI == 'true' &&
    process.platform == 'linux' &&
    process.arch == 'arm64') {
  mx.disableCompile();
}

const {values} = util.parseArgs({
  options: {
    grep: {type: 'string', short: 'g'},
    invert: {type: 'string', short: 'i'},
  }
});

const mocha = new Mocha();
if (values.grep) mocha.grep(values.grep);
if (values.invert) mocha.invert();

for (const f of fs.readdirSync(__dirname)) {
  if (f.endsWith('.spec.ts') || f.endsWith('.spec.js'))
    mocha.addFile(path.join(__dirname, f));
}
mocha.run(process.exit);
