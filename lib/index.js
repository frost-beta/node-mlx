// Default exports is the native binding.
module.exports = require(__dirname + '/../build/Release/node-mlx.node');

// The stream helper is implemented in JS.
module.exports.stream = require('../dist/stream').stream;
