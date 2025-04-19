import * as nn from '../nn';
import * as utils from '../utils';
import {core as mx} from '../core';
import {Nested, NestedDict, treeMap} from '../utils';

/**
 * The base class for all optimizers. It allows us to implement an optimizer on
 * a per-parameter basis and apply it to a parameter tree.
 */
export abstract class Optimizer {
  #initialized = false;
  #state: Record<string, unknown> = {step: mx.array(0, mx.uint64)};
  #schedulers: Record<string, (step: mx.array) => mx.array>;

  /**
   * Create an optimizer instance.
   *
   * @param schedulers - An optional dictionary of schedulers.
   */
  constructor(schedulers: Record<string, (step: mx.array) => mx.array> = {}) {
    this.#schedulers = schedulers;
  }

  /**
   * Apply the gradients to the parameters of the model and update the model
   * with the new parameters.
   *
   * @param model - An MLX module to be updated.
   * @param gradients - A tree of gradients, most likely computed via
   * `mlx.nn.valueAndGrad`.
   */
  update(model: nn.Module, gradients: Nested<mx.array>): void {
    model.update(this.applyGradients(gradients, model.parameters()) as NestedDict<mx.array>);
  }

  /**
   * Initialize the optimizer's state.
   *
   * @remarks
   *
   * This function can be used to initialize optimizers which have state (like
   * momentum in `SGD`). Using this method is optional as the optimizer will
   * initialize itself if the state is not yet set. However, there are some
   * cases where explicit initialization is useful in order to have access to
   * the `state` before the first call to `update`.
   *
   * @param parameters - A tree of parameters.
   *
   * @example
   * ```typescript
   * const optimizer = new optim.SGD({learningRate: 1e-1, momentum: 0.9});
   * const model = new nn.Linear(2, 2);
   * optimizer.init(model.trainableParameters());
   * console.log(Object.keys(optimizer.state)); // ['step', 'learningRate', 'weight', 'bias']
   * ```
   */
  init(parameters: Nested<mx.array>): void {
    // Iniatilize the optimizer state to match the parameter state.
    function updateState(params: Nested<mx.array>, state: Record<string, any>) {
      if (Array.isArray(params) && Array.isArray(state)) {
        for (let i = 0; i < state.length; ++i)
          state[i] = updateState(params[i], state[i]);
        if (state.length !== params.length)
          state.push(...<unknown[]>treeMap(x => ({}), params.slice(state.length)));
      } else if (typeof params === 'object') {
        for (const [k, v] of Object.entries(params)) {
          if (!state.hasOwnProperty(k))
            state[k] = treeMap(x => ({}), v);
          else
            state[k] = updateState(v, state[k]);
        }
      }
      return state;
    }
    updateState(parameters, this.state);
    treeMap((p, s) => Object.keys(s).length === 0 ? this.initSingle(p, s) : s, parameters, [ this.state ]);
    this.#initialized = true;
  }

  /**
   * Initialize a single parameter of the optimizer's state.
   *
   * @remarks
   *
   * To be overridden by subclasses to implement each optimizer's state
   * initialization.
   *
   * @param parameter - A single parameter that will be optimized.
   */
  abstract initSingle(parameter: mx.array,
                      state: Record<string, unknown>): void;

  /**
   * Apply an optimizer's update for a single parameter.
   *
   * @remarks
   *
   * To be overridden in subclasses to implement the optimizer's update.
   *
   * @param gradient - The parameter gradient.
   * @param parameter - The parameter to update.
   * @param state - The optimizer's state.
   */
  abstract applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, unknown>): unknown;

  /**
   * Apply the gradients to the parameters and return the updated parameters.
   *
   * @remarks
   *
   * Can be used to update a model via `model.update(opt.applyGradients(grads, model))`
   * which is precisely how `update` is implemented.
   *
   * @param gradients - A Python tree of gradients.
   * @param parameters - A Python tree of parameters. It can be a superset of
   * the gradients. In that case the returned python tree will be of the same
   * structure as the gradients.
   */
  applyGradients(gradients: Nested<mx.array>,
                 parameters: Nested<mx.array>): Nested<mx.array> {
    if (!this.#initialized)
      this.init(gradients);

    // Update any scheduled variables.
    for (let param in this.#schedulers) {
      mx.dispose(this.state[param]);
      this.state[param] = this.#schedulers[param](this.step);
    }

    // Increment the step.
    const step = mx.add(this.step, 1);
    mx.dispose(this.state['step']);
    this.state['step'] = step;

    // Apply the update.
    return utils.treeMap(this.applySingle.bind(this), gradients, [parameters, this.state]) as Nested<mx.array>;
  }

  /**
   * The optimizer's state dictionary.
   */
  get state(): Record<string, unknown> {
    return this.#state;
  }

  set state(state: Record<string, unknown>) {
    this.#initialized = false;
    this.#state = state;
  }

  get step(): mx.array {
    return this.state['step'] as mx.array;
  }

  get learningRate(): mx.array {
    return this.state['learningRate'] as mx.array;
  }

  set learningRate(learningRate: number | mx.array) {
    this.state['learningRate'] = mx.array(learningRate);
  }

  /*
   * To be used by derived classes to optionally put a parameter on a schedule.
   */
  protected maybeSchedule(name: string,
                          schedule: number | ((step: mx.array) => mx.array)): void {
    let param: mx.array;
    if (typeof schedule === 'function') {
      this.#schedulers[name] = schedule;
      param = schedule(this.step);
    } else {
      param = mx.array(schedule);
    }
    this.state[name] = param;
  }
}

/**
 * Wraps a list of optimizers with corresponding weight predicates/filters
 * to make it easy to use different optimizers for different weights.
 *
 * The predicates take the full "path" of the weight and the weight itself and
 * return `true` if it should be considered for this optimizer. The last
 * optimizer in the list is a fallback optimizer and no predicate should be
 * given for it.
 */
export class MultiOptimizer extends Optimizer {
  #optimizers: Optimizer[];
  #filters: ((path: string, weight: mx.array) => boolean)[];

  /**
   * Create a `MultiOptimizer` instance.
   *
   * @param optimizers - A list of optimizers to delegate to.
   * @param filters - A list of predicates that should be one less than the
   * provided optimizers.
   */
  constructor(optimizers: Optimizer[],
              filters: ((path: string, weight: mx.array) => boolean)[] = []) {
    super();

    if (filters.length !== optimizers.length - 1)
      throw new Error(`Given ${filters.length} filters but ${optimizers.length - 1} needed.`);

    this.#optimizers = optimizers;
    this.#filters = filters.concat([(k: string, g: mx.array) => true]);
  }

  override init(parameters: NestedDict<mx.array>) {
    for (let i in this.#optimizers)
      this.#optimizers[i].init(this.splitDictionary(parameters)[i]);
  }

  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    throw new Error('MultiOptimizer does not implement initSingle');
  }

  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    throw new Error('MultiOptimizer does not implement applySingle');
  }

  override applyGradients(gradients: NestedDict<mx.array>, parameters: Nested<mx.array>): NestedDict<mx.array> {
    let tree: NestedDict<mx.array> = {};
    for (let i in this.#optimizers) {
      const o = this.#optimizers[i];
      const g = this.splitDictionary(gradients)[i];
      tree = utils.treeMerge(tree, o.applyGradients(g, parameters)) as NestedDict<mx.array>;
    }
    return tree;
  }

  override get state(): Record<string, unknown> {
    return {states: this.#optimizers.map(o => o.state)};
  }

  override set state(state: Record<string, unknown>) {
    if (!state.hasOwnProperty('states') || (<unknown[]>state['states']).length !== this.#optimizers.length)
      throw new Error('Invalid state provided');
    for (let i in this.#optimizers)
      this.#optimizers[i].state = <Record<string, unknown>>(<unknown[]>state['states'])[i];
  }

  override get learningRate(): mx.array {
    return this.#optimizers[0].learningRate;
  }

  override set learningRate(learningRate: number | mx.array) {
    for (let o of this.#optimizers)
      o.learningRate = learningRate;
  }

  private splitDictionary(gradients: NestedDict<mx.array>): NestedDict<mx.array>[] {
    if (this.#optimizers.length === 1)
      return [gradients];

    const parts: Array<[string, mx.array]>[] = [];
    for (let i = 0; i < this.#optimizers.length; ++i)
      parts.push([]);
    for (let [k, g] of utils.treeFlatten(gradients)) {
      for (let i in this.#filters) {
        if (this.#filters[i](k, g as mx.array)) {
          parts[i].push([k, g as mx.array]);
          break;
        }
      }
    }

    return parts.map(p => utils.treeUnflatten(p) as NestedDict<mx.array>);
  }
}

/**
 * The stochastic gradient descent optimizer.
 *
 * @remarks
 *
 * Updates a parameter `w` with a gradient `g` as follows
 *
 * ```math
 * v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
 * w_{t+1} &= w_t - \lambda v_{t+1}
 * ```
 *
 * @param learningRate - The learning rate `\lambda`.
 * @param momentum - The momentum strength `\mu`. Default: `0`.
 * @param weightDecay - The weight decay (L2 penalty). Default: `0`.
 * @param dampening - Dampening for momentum `\tau`. Default: `0`.
 * @param nesterov - Enables Nesterov momentum. Default: `False`.
 */
export class SGD extends Optimizer {
  momentum: number;
  weightDecay: number;
  dampening: number;
  nesterov: boolean;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              momentum: number = 0,
              weightDecay: number = 0,
              dampening: number = 0,
              nesterov: boolean = false) {
    if (nesterov && (momentum <= 0 || dampening != 0))
      throw new Error('Nesterov momentum requires a momentum and zero dampening.');
    super();

    this.maybeSchedule('learningRate', learningRate);
    this.momentum = momentum;
    this.weightDecay = weightDecay;
    this.dampening = dampening;
    this.nesterov = nesterov;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['v'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the SGD parameter update and stores `v` in the
   * optimizer state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    if (this.weightDecay !== 0)
      gradient = mx.add(gradient, mx.multiply(this.weightDecay, parameter));

    if (this.momentum <= 0) {
      return mx.subtract(parameter,
                         mx.multiply(this.learningRate.astype(gradient.dtype),
                                     gradient));
    }

    let v = mx.multiply(this.momentum, state['v']);
    if (this.dampening > 0)
      v = mx.add(v, mx.multiply(1 - this.dampening, gradient));
    else
      v = mx.add(v, gradient);

    let update;
    if (this.nesterov)
      update = mx.add(gradient, mx.multiply(this.momentum, v));
    else
      update = v;

    state['v'] = v;
    return mx.subtract(parameter,
                       mx.multiply(this.learningRate.astype(gradient.dtype),
                                   gradient));
  }
}

/**
 * The RMSprop optimizer.
 *
 * @remarks
 *
 * Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural
 * networks for machine learning
 *
 * ```math
 * v_{t+1} &= \alpha v_t + (1 - \alpha) g_t^2 \\
 * w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}
 * ```
 *
 * @param learningRate - The learning rate.
 * @param alpha - The smoothing constant `\alpha`. Default: `0.99`.
 * @param eps - The term `\epsilon` added to the denominator to improve
 * numerical stability. Default: `1e-8`.
 */
export class RMSprop extends Optimizer {
  alpha: number;
  eps: number;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              alpha: number = 0.99,
              eps: number = 1e-8) {
    if (alpha < 0.0)
      throw new Error(`RMSprop alpha should be >=0, ${alpha} was provided instead`);
    if (eps <= 0.0)
      throw new Error(`RMSprop epsilon should be >0, ${eps} was provided instead`);
    super();

    this.maybeSchedule('learningRate', learningRate);
    this.alpha = alpha;
    this.eps = eps;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['v'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the RMSprop parameter update and stores `v` in the optimizer state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    const alpha = this.alpha;
    const eps = this.eps;

    let v = state['v'];
    v = mx.add(mx.multiply(alpha, v),
               mx.multiply(1 - alpha, mx.square(gradient)));

    mx.dispose(state['v']);
    state['v'] = v;

    return mx.subtract(parameter,
                       mx.multiply(lr,
                                   mx.divide(gradient,
                                             mx.add(mx.sqrt(v), eps))));
  }
}

/**
 * The Adagrad optimizer.
 *
 * @remarks
 *
 * Our Adagrad implementation follows the original paper. In detail,
 *
 * Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods
 * for online learning and stochastic optimization. JMLR 2011.
 *
 * ```math
 * v_{t+1} &= v_t + g_t^2 \\
 * w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}
 * ```
 *
 * @param learningRate - The learning rate `\lambda`.
 * @param eps - The term `\epsilon` added to the denominator to improve
 * numerical stability. Default: `1e-8`.
 */
export class Adagrad extends Optimizer {
  eps: number;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              eps: number = 1e-8) {
    if (eps < 0.0)
      throw new Error(`Adagrad epsilon should be >0, ${eps} was provided instead`);
    super();

    this.maybeSchedule('learningRate', learningRate);
    this.eps = eps;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['v'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the Adagrad parameter update and stores `v` in the
   * optimizer state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    const eps = this.eps;

    const v = mx.add(state['v'], mx.square(gradient));
    mx.dispose(state['v']);
    state['v'] = v;

    return mx.subtract(parameter,
                       mx.divide(mx.multiply(lr, gradient),
                                 mx.add(mx.sqrt(v), eps)));
  }
}

/**
 * The AdaDelta optimizer with a learning rate [1].
 *
 * @remarks
 *
 * Our AdaDelta implementation follows the original paper. In detail,
 *
 * Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method.
 * arXiv preprint arXiv:1212.5701.
 *
 * ```math
 * v_{t+1} &= \rho v_t + (1 - \rho) g_t^2 \\
 * \Delta w_{t+1} &= \frac{\sqrt{u_t + \epsilon}}{\sqrt{v_{t+1} + \epsilon}} g_t \\
 * u_{t+1} &= \rho u_t + (1 - \rho) \Delta w_{t+1}^2 \\
 * w_{t+1} &= w_t - \lambda \Delta w_{t+1}
 * ```
 *
 * @param learningRate - The learning rate `\lambda`.
 * @param rho - The coefficient `\rho` used for computing a running average of
 * squared gradients. Default: `0.9`.
 * @param eps - The term `\epsilon` added to the denominator to improve
 * numerical stability. Default: `1e-8`.
 */
export class AdaDelta extends Optimizer {
  rho: number;
  eps: number;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              rho: number = 0.9,
              eps: number = 1e-6) {
    if (rho < 0.0)
      throw new Error(`AdaDelta rho should be >=0, ${rho} was provided instead`);
    if (eps < 0.0)
      throw new Error(`AdaDelta epsilon should be >0, ${eps} was provided instead`);
    super();

    this.maybeSchedule('learningRate', learningRate);
    this.rho = rho;
    this.eps = eps;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['v'] = mx.zerosLike(parameter);
    state['u'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the AdaDelta parameter update and stores `v` and `u` in the
   * optimizer state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    const rho = this.rho;
    const eps = this.eps;

    let v = state['v'];
    let u = state['u'];

    v = mx.add(mx.multiply(rho, v),
               mx.multiply(1 - rho, mx.square(gradient)));
    const d = mx.multiply(mx.divide(mx.sqrt(mx.add(u, eps)),
                                    mx.sqrt(mx.add(v, eps))),
                          gradient);
    u = mx.add(mx.multiply(rho, u),
               mx.multiply(1 - rho, mx.square(d)));

    mx.dispose(state['v']);
    mx.dispose(state['u']);
    state['v'] = v;
    state['u'] = u;

    return mx.subtract(parameter, mx.multiply(lr, d));
  }
}

/**
 * The Adam optimizer.
 *
 * @remarks
 *
 * In detail,
 *
 * ```math
 * m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
 * v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
 * w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}}
 * ```
 *
 * @param learningRate - The learning rate `\lambda`.
 * @param betas - The coefficients `(\beta_1, \beta_2)` used for computing
 * running averages of the gradient and its square. Default: `[0.9, 0.999]`
 * @param eps - The term `\epsilon` added to the denominator to improve
 * numerical stability. Default: ``1e-8``
 * @param biasCorrection - If set to `True`, bias correction is applied.
 * Default: `False`.
 */
export class Adam extends Optimizer {
  betas: number[];
  eps: number;
  biasCorrection: boolean;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              betas: [number, number] = [0.9, 0.999],
              eps = 1e-8,
              biasCorrection = false) {
    super();

    this.maybeSchedule('learningRate', learningRate);
    this.betas = betas;
    this.eps = eps;
    this.biasCorrection = biasCorrection;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['m'] = mx.zerosLike(parameter);
    state['v'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the Adam parameter update and stores `v` and `m` in the optimizer
   * state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    const [b1, b2] = this.betas;
    const eps = this.eps;
    const biasCorrection = this.biasCorrection;
    const step = this.step;

    const m = mx.add(mx.multiply(b1, state['m']),
                     mx.multiply(1 - b1, gradient));
    const v = mx.add(mx.multiply(b2, state['v']),
                     mx.multiply(1 - b2, mx.square(gradient)));

    mx.dispose(state['m']);
    mx.dispose(state['v']);
    state['m'] = m;
    state['v'] = v;

    if (biasCorrection) {
      const numerator = mx.multiply(mx.divide(lr,
                                              mx.subtract(1,
                                                          mx.power(b1, step))),
                                    m);
      const denominator = mx.add(mx.divide(mx.sqrt(v),
                                           mx.sqrt(mx.subtract(1,
                                                               mx.power(b2, step)))),
                                 eps);
      return mx.subtract(parameter,
                         mx.divide(numerator, denominator));
    } else {
      return mx.subtract(parameter,
                         mx.divide(mx.multiply(lr, m),
                                   mx.add(mx.sqrt(v), eps)));
    }
  }
}

/**
 * The AdamW optimizer.
 *
 * @remarks
 *
 * We update the weights with a weight_decay (:math:`\lambda`) value:
 *
 * ```math
 * m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
 * v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
 * w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}} + \lambda w_t)
 * ```
 *
 * @param learningRate - The learning rate `\alpha`.
 * @param betas - The coefficients `(\beta_1, \beta_2)` used for computing
 * running averages of the gradient and its square. Default: `[0.9, 0.999]`.
 * @param eps - The term `\epsilon` added to the denominator to improve
 * numerical stability. Default: `1e-8`.
 * @param weightDecay - The weight decay `\lambda`. Default: `0`.
 * @param biasCorrection - If set to `True`, bias correction is applied.
 * Default: `False`.
 */
export class AdamW extends Adam {
  weightDecay: number;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              betas: [number, number] = [0.9, 0.999],
              eps = 1e-8,
              weightDecay = 0.01,
              biasCorrection = false) {
    super(learningRate, betas, eps, biasCorrection);
    this.weightDecay = weightDecay;
  }

  /**
   * Performs the AdamW parameter update by modifying the parameters passed into
   * Adam.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    return super.applySingle(gradient,
                             mx.multiply(parameter,
                                         mx.subtract(1, mx.multiply(lr, this.weightDecay))),
                             state);
  }
}

/**
 * The Adamax optimizer, a variant of Adam based on the infinity norm.
 *
 * @remarks
 *
 * Our Adam implementation follows the original paper and omits the bias
 * correction in the first and second moment estimates. In detail,
 *
 * Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
 * optimization. ICLR 2015.
 *
 * ```math
 * m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
 * v_{t+1} &= \max(\beta_2 v_t, |g_t|) \\
 * w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{v_{t+1} + \epsilon}
 * ```
 *
 * @param learningRate - The learning rate `\lambda`.
 * @param betas - The coefficients `(\beta_1, \beta_2)` used for computing
 * running averages of the gradient and its square. Default: `[0.9, 0.999]`.
 * @param eps - The term `\epsilon` added to the denominator to improve
 * numerical stability. Default: `1e-8`.
 */
export class Adamax extends Adam {
  constructor(learningRate: number | ((step: mx.array) => mx.array),
              betas: [number, number] = [0.9, 0.999],
              eps = 1e-8) {
    if (eps < 0)
      throw new Error(`Epsilon value should be >=0, ${eps} was provided instead`);
    super(learningRate, betas, eps);
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['m'] = mx.zerosLike(parameter);
    state['v'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the Adamax parameter update and stores `m` and `v` in the
   * optimizer state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    const [b1, b2] = this.betas;
    const eps = this.eps;

    const m = mx.add(mx.multiply(b1, state['m']),
                     mx.multiply(1 - b1, gradient));
    const v = mx.maximum(mx.multiply(b2, state['v']),
                         gradient.abs());

    mx.dispose(state['m']);
    mx.dispose(state['v']);
    state['m'] = m;
    state['v'] = v;

    return mx.subtract(parameter,
                       mx.divide(mx.multiply(lr, m),
                                 mx.add(v, eps)));
  }
}

/**
 * The Lion optimizer.
 *
 * @remarks
 *
 * Since updates are computed through the sign operation, they tend to have
 * larger norm than for other optimizers such as SGD and Adam. We recommend a
 * learning rate that is 3-10x smaller than AdamW and a weight decay 3-10x
 * larger than AdamW to maintain the strength (lr * wd). Our Lion implementation
 * follows the original paper. In detail,
 *
 * Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv preprint
 * arXiv:2302.06675.
 *
 * ```math
 * c_{t + 1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
 * m_{t + 1} &= \beta_2 m_t + (1 - \beta_2) g_t \\
 * w_{t + 1} &= w_t - \eta (\text{sign}(c_t) + \lambda w_t)
 * ```
 *
 * @param learningRate - The learning rate `\eta`.
 * @param betas - The coefficients `(\beta_1, \beta_2)` used for computing the
 * gradient momentum and update direction. Default: `[0.9, 0.99]`.
 * @param weightDecay - The weight decay `\lambda`. Default: `0.0`.
 */
export class Lion extends Optimizer {
  betas: number[];
  weightDecay: number;

  constructor(learningRate: number | ((step: mx.array) => mx.array),
              betas: [number, number] = [0.9, 0.99],
              weightDecay = 0) {
    super();

    this.maybeSchedule('learningRate', learningRate);
    this.betas = betas;
    this.weightDecay = weightDecay;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    state['m'] = mx.zerosLike(parameter);
  }

  /**
   * Performs the Lion parameter update and stores `m` in the optimizer state.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const lr = this.learningRate.astype(gradient.dtype);
    const [b1, b2] = this.betas;

    const m = state['m'];
    const c = mx.add(mx.multiply(b1, m),
                     mx.multiply(1 - b1, gradient));
    const d = mx.add(mx.multiply(b2, m),
                     mx.multiply(1 - b2, gradient));
    mx.dispose(state['m']);
    state['m'] = d;
    if (this.weightDecay > 0) {
      parameter = mx.multiply(mx.subtract(1, mx.multiply(lr, this.weightDecay)),
                              parameter);
    }
    return mx.subtract(parameter,
                       mx.multiply(lr, mx.sign(c)));
  }
}

/**
 * The Adafactor optimizer.
 *
 * @remarks
 *
 * Our Adafactor implementation follows the original paper:
 * Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
 * https://arxiv.org/abs/1804.04235
 *
 * @param learningRate - The learning rate.
 * @param eps - The first term `\epsilon_1` added to the square of the gradients
 * to improve numerical stability and the second term `\epsilon_2` is used for
 * parameter scaling if `parameterScale` is set to `True`. Default: `[1e-30,
 * 1e-3]`.
 * @param clipThreshold - Clips the unscaled update at `clipThreshold`. Default:
 * `1.0`.
 * @param decayRate - Coefficient for the running average of the squared
 * gradient. Default: `-0.8`.
 * @param beta1 - If set to a value bigger than zero then first moment will be
 * used. Default: `None`.
 * @param weightDecay - The weight decay `\lambda`. Default: `0.0`.
 * @param scaleParameter If set to `True` the learning rate will be scaled by
 * `\max(\epsilon_1, \text{RMS}(w_{t-1}))`. Default: `True`.
 * @param relativeStep If set to `True` the `learningRate` will be ignored and
 * relative step size will be computed. Default: `True`.
 * @param warmupInit If set to `True` then the relative step size will be
 * calculated by the current step. Default: `False`.
 */
export class Adafactor extends Optimizer {
  eps: [number, number];
  clipThreshold: number;
  decayRate: number;
  beta1: number | null;
  weightDecay: number;
  scaleParameter: boolean;
  relativeStep: boolean;
  warmupInit: boolean;

  constructor(learningRate: number | ((step: mx.array) => mx.array) | null = null,
              eps: [number, number] = [1e-30, 1e-3],
              clipThreshold: number = 1.0,
              decayRate: number = -0.8,
              beta1: number | null = null,
              weightDecay = 0.0,
              scaleParameter = true,
              relativeStep = true,
              warmupInit = false) {
    super();
    if (learningRate !== null)
      this.maybeSchedule('learningRate', learningRate);
    this.eps = eps;
    this.clipThreshold = clipThreshold;
    this.decayRate = decayRate;
    this.beta1 = beta1;
    this.weightDecay = weightDecay;
    this.scaleParameter = scaleParameter;
    this.relativeStep = relativeStep;
    this.warmupInit = warmupInit;
  }

  /**
   * Initialize optimizer state.
   */
  override initSingle(parameter: mx.array,
                      state: Record<string, mx.array>) {
    if (parameter.ndim >= 2) {
      const shape = parameter.shape;
      const dtype = parameter.dtype;
      state['expAvgSqRow'] = mx.zeros(shape.slice(0, -1), dtype);
      state['expAvgSqCol'] = mx.zeros(shape.slice(0, -2).concat(shape.slice(-1)), dtype);
    } else {
      state['expAvgSq'] = mx.zerosLike(parameter);
    }

    if (this.beta1) {
      state['expAvg'] = mx.zerosLike(parameter);
    }
  }

  /**
   * Performs the Adafactor parameter and state update.
   */
  override applySingle(gradient: mx.array,
                       parameter: mx.array,
                       state: Record<string, mx.array>) {
    const factored = gradient.ndim >= 2;

    const step = this.step;
    const useFirstMoment = this.beta1 != null;

    const parameterRMS = this.computeRMS(parameter);
    const learningRate = this.computeLearningRate(step, parameterRMS);
    const beta2 = mx.subtract(1, mx.power(step, this.decayRate))
                    .astype(parameterRMS.dtype);

    let update = mx.add(mx.square(gradient),
                        mx.array(this.eps[0], gradient.dtype));

    if (factored) {
      let expAvgSqRow = state['expAvgSqRow'];
      let expAvgSqCol = state['expAvgSqCol'];
      expAvgSqRow = mx.add(mx.multiply(beta2, expAvgSqRow),
                           mx.multiply(mx.subtract(1, beta2),
                                       mx.mean(update, -1)));
      expAvgSqCol = mx.add(mx.multiply(beta2, expAvgSqCol),
                           mx.multiply(mx.subtract(1, beta2),
                                       mx.mean(update, -2)));
      mx.dispose(state['expAvgSqRow']);
      mx.dispose(state['expAvgSqCol']);
      state['expAvgSqRow'] = expAvgSqRow;
      state['expAvgSqCol'] = expAvgSqCol;
      update = this.approximateExpMovingAvg(expAvgSqRow, expAvgSqCol);
      update = mx.multiply(update, gradient);
    } else {
      let expAvgSq = state['expAvgSq'];
      expAvgSq = mx.add(mx.multiply(beta2, expAvgSq),
                        mx.multiply(mx.subtract(1, beta2),
                                    update));
      mx.dispose(state['expAvgSq']);
      state['expAvgSq'] = expAvgSq;
      update = mx.multiply(mx.rsqrt(expAvgSq), gradient);
    }

    update = mx.divide(update,
                       mx.maximum(1,
                                  mx.divide(this.computeRMS(update),
                                            mx.array(this.clipThreshold, update.dtype))));
    update = mx.multiply(learningRate, update);

    if (useFirstMoment) {
      let expAvg = state['expAvg'];
      expAvg = mx.add(mx.multiply(this.beta1, expAvg),
                      mx.multiply(1 - this.beta1, update));
      mx.dispose(state['expAvg']);
      state['expAvg'] = expAvg;
      update = expAvg;
    }

    if (this.weightDecay != 0) {
      parameter = mx.add(parameter,
                         mx.multiply(parameter,
                                     mx.multiply(learningRate,
                                                 -this.weightDecay)));
    }

    return mx.subtract(parameter, update);
  }

  private computeRMS(input: mx.array): mx.array {
    return mx.sqrt(mx.mean(mx.square(input)));
  }

  private computeLearningRate(step: mx.array,
                              parameterRMS: mx.array): mx.array {
    let relativeStepSize;
    if (this.relativeStep) {
      const minStep = this.warmupInit? mx.multiply(1e-6, step) : 1e-2;
      relativeStepSize = mx.minimum(minStep, mx.rsqrt(step));
    } else {
      relativeStepSize = this.learningRate;
    }

    relativeStepSize = relativeStepSize.astype(parameterRMS.dtype);
    let parameterScale = mx.array(1, parameterRMS.dtype);
    if (this.scaleParameter) {
      parameterScale = mx.maximum(mx.array(this.eps[1], parameterRMS.dtype),
                                  parameterRMS);
    }
    return mx.multiply(parameterScale, relativeStepSize);
  }

  private approximateExpMovingAvg(expAvgSqRow: mx.array,
                                  expAvgSqCol: mx.array): mx.array {
    const rFactor = mx.rsqrt(mx.divide(expAvgSqRow,
                                       mx.mean(expAvgSqRow, -1, true)));
    const cFactor = mx.rsqrt(expAvgSqCol);
    return mx.matmul(mx.expandDims(rFactor, -1),
                     mx.expandDims(cFactor, 0));
  }
}

/**
 * Clips the global norm of the gradients.
 *
 * @remarks
 *
 * This function ensures that the global norm of the gradients does not exceed
 * `maxNorm`. It scales down the gradients proportionally if their norm is
 * greater than `maxNorm`.
 *
 * Example:
 * ```typescript
 * const grads = {'w1': mx.array([2, 3]), 'w2': mx.array([1])};
 * const [clippedGrads, totalNorm] = clipGradNorm(grads, 2.0);
 * console.log(clippedGrads);
 * // {"w1": mx.array([...]), "w2": mx.array([...])}
 * ```
 *
 * @param grads - A dictionary containing the gradient arrays.
 * @param maxNorm - The maximum allowed global norm of the gradients.
 * @returns The possibly rescaled gradients and the original gradient norm.
 */
export function clipGradNorm(grads: Nested<mx.array>,
                             maxNorm: number): [Nested<mx.array>, mx.array] {
  const normSquared = utils.treeReduce((acc: number | mx.array, g: mx.array) => {
    return mx.add(acc, g.square().sum());
  }, grads, 0);
  const totalNorm = mx.sqrt(normSquared);
  const normalizer = mx.divide(maxNorm, mx.add(totalNorm, 1e-6));

  function clipper(g: mx.array) {
    return mx.where(mx.less(totalNorm, maxNorm), g, mx.multiply(g, normalizer));
  }

  const clippedGrads = utils.treeMap(clipper, grads) as Nested<mx.array>;
  return [clippedGrads, totalNorm];
}
