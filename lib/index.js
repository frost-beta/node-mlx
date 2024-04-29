// The core APIs are in native bindings.
const core = require(__dirname + '/../build/Release/node-mlx.node');

// Helper for creating complex number.
core.Complex = (re, im) => {
  return {re, im: im ?? 0};
};

// The stream helper is in TS.
core.stream = require('../dist/stream').stream;

module.exports = {core};

// The nn module is in JS, lazy-load it to avoid circular reference.
Object.defineProperty(module.exports, 'nn', {
  configurable: true,
  enumerable: true,
  get() {
    const nn = require('../dist/nn');
    Object.defineProperty(module.exports, 'nn', {
      value: nn,
      writable: false,
      configurable: false,
    });
    return nn;
  }
});
