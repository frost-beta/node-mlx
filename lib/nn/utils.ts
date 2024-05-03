import {core as mx} from '../..';
import {Module} from './layers/base';

/**
 * Transform the passed function `func` to a function that computes the
 * gradients of `func` with respect to the model's trainable parameters and also its
 * value.
 *
 * @param model The model whose trainable parameters to compute
 * gradients for
 * @param func The scalar function to compute gradients for
 * @returns A callable that returns the value of `func` and the gradients with respect to the
 * trainable parameters of `model`
 */
export function valueAndGrad<T extends any[], U>(model: Module,
                                                 func: (...args: T) => U) {
  const innerFn = (params: Record<string, unknown>, ...args: T) => {
    model.update(params);
    return func(...args);
  }

  const valueGradFn = mx.valueAndGrad(innerFn);

  const wrappedValueGradFn = (...args: T) => {
    let [value, grad] = valueGradFn(model.trainableParameters(), ...args);
    return [value, grad];
  }

  return wrappedValueGradFn;
}
