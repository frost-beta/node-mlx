/// <reference path="../node_mlx.node.d.ts" preserve="true"/>
import core from '../build/Release/node_mlx.node';

// Helper for creating complex number.
core.Complex = (re, im) => {
  return { re, im: im ?? 0 };
};

// Implementation of the StreamContext.
core.stream = function stream(s: core.StreamOrDevice): Disposable {
  const old = core.defaultStream(core.defaultDevice());
  const target = core.toStream(s);
  core.setDefaultDevice(target.device);
  core.setDefaultStream(target);
  return {
    [Symbol.dispose]() {
      core.setDefaultDevice(old.device);
      core.setDefaultStream(old);
    },
  };
};

export { core, core as mx };
