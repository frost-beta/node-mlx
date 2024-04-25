import fs from 'node:fs';
import path from 'node:path';
import Mocha from 'mocha';
import yargs from 'yargs';
import tf from '@tensorflow/tfjs';

// Do not warn about using pure-js tensorflow.
tf.ENV.set('IS_TEST', true);

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
