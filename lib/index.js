// The core APIs are in native bindings.
const core = require(__dirname + '/../build/Release/mlx.node');

// Helper for creating complex number.
core.Complex = (re, im) => {
  return {re, im: im ?? 0};
};

// The stream helper is in TS.
core.stream = require('../dist/stream').stream;

// Export core (mx) module.
exports.core = core;

// Export utils module.
exports.utils = require('../dist/utils');

// Lazy-load nn/optimizers modules to avoid circular references.
const cache = {};
defineLazyModule('nn');
defineLazyModule('optimizers');

// Export nn/optimizers modules.
// Note that while it is tempting to get rid of the |cache| object and define
// the lazy property on |exports| directly, doing so would make the exports
// undetectable from cjs-module-lexer, and "import {core} from 'mlx'" will not
// work in Node.js.
Object.defineProperty(exports, 'nn', {
  enumerable: true,
  get() { return cache.nn; }
});
Object.defineProperty(exports, 'optimizers', {
  enumerable: true,
  get() { return cache.optimizers; }
});

// Helper to define a lazy loaded property on |cache|.
function defineLazyModule(name) {
  Object.defineProperty(cache, name, {
    configurable: true,
    enumerable: true,
    get() {
      const mod = require(`../dist/${name}`);
      Object.defineProperty(cache, name, {
        value: mod,
        writable: false,
        configurable: false,
      });
      return mod;
    }
  });
}
