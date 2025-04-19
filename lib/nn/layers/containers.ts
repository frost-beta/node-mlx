import {core as mx} from '../../core';
import {Module} from './base';

interface SequentialModule extends Module {
  forward(x: mx.array): mx.array;
}

/**
 * A layer that calls the passed callables in order.
 *
 * @remarks
 *
 * We can pass either modules or plain callables to the Sequential module. If
 * our functions have learnable parameters they should be implemented as
 * `nn.Module` instances.
 *
 * @param modules The modules to call in order
 */
export class Sequential extends Module {
  layers: (SequentialModule | ((x: mx.array) => mx.array))[];

  constructor(...modules: (SequentialModule | ((x: mx.array) => mx.array))[]) {
    super();
    this.layers = modules;
  }

  override forward(x: mx.array): mx.array {
    for (const layer of this.layers) {
      x = typeof layer === 'function' ? layer(x) : layer.forward(x);
    }
    return x;
  }
}
