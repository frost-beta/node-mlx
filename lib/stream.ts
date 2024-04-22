import {StreamOrDevice} from '..';

// Implementation of the StreamContext in JS.
export function stream(s: StreamOrDevice): Disposable {
  const mx = require('..');  // work around circular import
  const old = mx.defaultStream(mx.defaultDevice());
  const target = mx.toStream(s);
  mx.setDefaultDevice(target.device);
  mx.setDefaultStream(target);
  return {
    [Symbol.dispose]() {
      mx.setDefaultDevice(old.device);
      mx.setDefaultStream(old);
    }
  };
}
