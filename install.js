#!/usr/bin/env node

const packageJson = require('./package.json');

// Local developement.
if (packageJson.version === '0.0.1-dev' && !process.argv.includes('--force'))
  process.exit(0);

const fs = require('node:fs');
const path = require('node:path');
const stream = require('node:stream');
const util = require('node:util');
const zlib = require('node:zlib');

const urlPrefix = 'https://github.com/frost-beta/node-mlx/releases/download';

main().catch((error) => {
  console.error('Error downloading node-mlx:', error);
  process.exit(1);
});

async function main() {
  const os = {darwin: 'mac', win32: 'win'}[process.platform] ?? process.platform;
  const arch = process.arch;
  const version = packageJson.version;
  const url = `${urlPrefix}/v${version}/mlx-${os}-${arch}.node.gz`;

  const outputDir = path.join(__dirname, 'build', 'Release');
  fs.mkdirSync(outputDir, {recursive: true});

  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`HTTP error! Status: ${response.status}`);

  const pipeline = promisify(stream.pipeline);
  const gunzip = zlib.createGunzip();
  await pipeline(response.body,
                 gunzip,
                 fs.createWriteStream(path.join(outputDir, 'mlx.node')));
}
