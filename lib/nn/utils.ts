import {core as mx} from '../..';
import {Module} from './layers/base';
import {NestedDict} from '../utils';

/**
 * Transform the passed function `func` to a function that computes the
 * gradients of `func` with respect to the model's trainable parameters and also
 * its value.
 *
 * @param model The model whose trainable parameters to compute gradients for.
 * @param func The scalar function to compute gradients for.
 * @returns A callable that returns the value of `func` and the gradients with
 * respect to the trainable parameters of `model`.
 */
export function valueAndGrad<T extends any[], U>(model: Module,
                                                 func: (...args: T) => U,
                                                 options?: {noAutoDispose: boolean}) {
  const innerFn = (params: NestedDict<mx.array>, ...args: T) => {
    model.update(params);
    return func(...args);
  }

  const valueGradFn = mx.valueAndGrad(innerFn);

  function wrappedValueGradFn(...args: T) {
    const params = model.trainableParameters();
    let [value, grad] = valueGradFn(params, ...args);
    // The mx.valueAndGrad API replaces the params with tracers and they got
    // updated into the model, so we can dispose the old params.
    if (!options.noAutoDispose)
      mx.dispose(params);
    return [value, grad];
  }

  return wrappedValueGradFn;
}

/**
 * Transform the passed callable to one that performs gradient checkpointing
 * with respect to the trainable parameters of the module (and the callable's
 * inputs).
 *
 * @param mod The module for whose parameters to perform gradient checkpointing.
 * @param func The function to checkpoint. If not provided, it defaults to the
 * provided module.
 * @returns A function that saves the inputs and outputs during the forward pass
 * and recomputes all intermediate states during the backward pass.
 */
export function checkpoint<M extends Module, T extends any[]>(
    mod: M,
    func?: (...args: T) => ReturnType<M['forward']>) {
  if (!func) {
    func = (...args: T) => mod.forward(...args) as ReturnType<M['forward']>;
  }

  const innerFn = (params: NestedDict<mx.array>, ...args: T) => {
    mod.update(params);
    return func(...args);
  };

  const checkpointedFn = mx.checkpoint(innerFn);

  function wrappedCheckpointedFn(...args: T) {
    return checkpointedFn(mod.trainableParameters(), ...args);
  }

  return wrappedCheckpointedFn;
}
