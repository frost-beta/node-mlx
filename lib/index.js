// The core APIs are in native bindings.
const core = require(__dirname + '/../build/Release/node-mlx.node');

// Helper for creating complex number.
core.Complex = (re, im) => {
  return {re, im: im ?? 0};
};

// The stream helper is in TS.
core.stream = require('../dist/stream').stream;

// The utils modules.
const utils = require('../dist/utils');

module.exports = {core, utils};

defineLazyModule('nn', '../dist/nn');
defineLazyModule('optimizers', '../dist/optimizers');

// Lazy-load modules to avoid circular references.
function defineLazyModule(name, path) {
  Object.defineProperty(module.exports, name, {
    configurable: true,
    enumerable: true,
    get() {
      const mod = require(path);
      Object.defineProperty(module.exports, name, {
        value: mod,
        writable: false,
        configurable: false,
      });
      return mod;
    }
  });
}
