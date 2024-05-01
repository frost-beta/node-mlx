import fs from 'node:fs';
import path from 'node:path';
import Mocha from 'mocha';
import yargs from 'yargs';
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

const argv = require('yargs')
  .string('g').alias('g', 'grep')
  .boolean('i').alias('i', 'invert')
  .argv;

const mocha = new Mocha();
if (argv.grep) mocha.grep(argv.grep);
if (argv.invert) mocha.invert();

for (const f of fs.readdirSync(__dirname)) {
  if (f.endsWith('.spec.ts'))
    mocha.addFile(path.join(__dirname, f));
}
mocha.run(process.exit);
