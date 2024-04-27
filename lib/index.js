// Default exports is the native binding.
module.exports = require(__dirname + '/../build/Release/node-mlx.node');

// Helper for creating complex number.
module.exports.Complex = (re, im) => {
  return {re, im: im ?? 0};
};

// The stream helper is implemented in JS.
module.exports.stream = require('../dist/stream').stream;
