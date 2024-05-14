#!/usr/bin/env node

const packageJson = require('./package.json');

// Local developement.
if (packageJson.version === '0.0.1-dev')
  process.exit(0);

const fs = require('node:fs');
const path = require('node:path');
const util = require('node:util');
const zlib = require('node:zlib');
const {pipeline} = require('node:stream/promises');

const urlPrefix = 'https://github.com/frost-beta/node-mlx/releases/download';

main().catch((error) => {
  console.error('Error downloading node-mlx:', error);
  process.exit(1);
});

async function main() {
  const dir = path.join(__dirname, 'build', 'Release');
  fs.mkdirSync(dir, {recursive: true});

  const os = {darwin: 'mac', win32: 'win'}[process.platform] ?? process.platform;
  const arch = process.arch;
  const version = packageJson.version;

  const prefix = `${urlPrefix}/v${version}/mlx-${os}-${arch}`;
  const tasks = [ download(`${prefix}.node.gz`, path.join(dir, 'mlx.node')) ];
  if (os == 'mac' && arch == 'arm64')
    tasks.push(download(`${prefix}.metallib.gz`, path.join(dir, 'mlx.metallib')));
  await Promise.all(tasks);
}

async function download(url, filename) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Failed to download ${url}, status: ${response.status}`);

  const gunzip = zlib.createGunzip();
  await pipeline(response.body, gunzip, fs.createWriteStream(filename));
}
