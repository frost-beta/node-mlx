// The core APIs are in native bindings.
const core = require(__dirname + '/../build/Release/node-mlx.node');

// Helper for creating complex number.
core.Complex = (re, im) => {
  return {re, im: im ?? 0};
};

// The stream helper is implemented in JS.
core.stream = require('../dist/stream').stream;

module.exports = {core};
