import {core as mx} from '../core';

/**
 * Make an exponential decay scheduler.
 *
 * @remarks
 *
 * @param init Initial value.
 * @param decayRate Multiplicative factor to decay by.
 *
 * @example
 * ```typescript
 * let lrSchedule = optim.exponentialDecay(0.1, 0.9);
 * let optimizer = new optim.SGD(lrSchedule);
 * console.log(optimizer.learningRate); // 0.1
 *
 * for (let i = 0; i < 5; i++) optimizer.update({}, {});
 * console.log(optimizer.learningRate); //0.06561
 * ```
 */
export function exponentialDecay(init: number, decayRate: number) {
  return (step: number | mx.array) => mx.multiply(init, mx.power(decayRate, step));
}

/**
 * Make a step decay scheduler.
 *
 * @remarks
 *
 * @param init Initial value.
 * @param decayRate Multiplicative factor to decay by.
 * @param stepSize Decay every `stepSize` steps.
 *
 * @example
 * ```typescript
 * let lrSchedule = optim.stepDecay(0.1, 0.9, 10);
 * let optimizer = new optim.SGD(lrSchedule);
 * console.log(optimizer.learningRate); // 0.1
 *
 * for (let i = 0; i < 21; i++) optimizer.update({}, {});
 * console.log(optimizer.learningRate); // 0.081
 * ```
 */
export function stepDecay(init: number, decayRate: number, stepSize: number) {
  return (step: number | mx.array) => mx.multiply(init, mx.power(decayRate, mx.floorDivide(step, stepSize)));
}

/**
 * Make a cosine decay scheduler.
 *
 * @remarks
 *
 * @param init Initial value.
 * @param decaySteps Number of steps to decay over. The decayed
 * value is constant for steps beyond `decaySteps`.
 * @param end Final value to decay to. Default: `0`.
 *
 * @example
 * ```typescript
 * let lrSchedule = optim.cosineDecay(0.1, 1000);
 * let optimizer = new optim.SGD(lrSchedule);
 * console.log(optimizer.learningRate); // 0.1
 *
 * for (let i = 0; i < 5; i++) optimizer.update({}, {});
 * console.log(optimizer.learningRate); // 0.0999961
 * ```
 */
export function cosineDecay(init: number, decaySteps: number, end = 0) {
  return (step: number | mx.array) => {
    let s = mx.minimum(step, decaySteps);
    let decay = mx.multiply(0.5,
                            mx.add(1, mx.cos(mx.multiply(mx.divide(Math.PI,
                                                                   decaySteps),
                                                         s))));
    return mx.add(end, mx.multiply(decay, init - end));
  };
}

/**
 * Join multiple schedules to create a new schedule.
 *
 * @param schedules A list of schedules. Schedule `i+1` receives a step count
 * indicating the number of steps since the `i`-th boundary.
 * @param boundaries A list of integers of length `schedules.length - 1` that indicates when
 * to transition between schedules.
 *
 * @example
 * ```typescript
 * let warmup = optim.linearSchedule(0, 0.1, 10);
 * let cosine = optim.cosineDecay(0.1, 200);
 * let lrSchedule = optim.joinSchedules([warmup, cosine], [10]);
 * let optimizer = new optim.Adam(lrSchedule);
 * console.log(optimizer.learningRate); // 0
 *
 * for (let i = 0; i < 12; i++) optimizer.update({}, {});
 * console.log(optimizer.learningRate); // 0.0999938
 * ```
 */
export function joinSchedules(schedules: ((step: number | mx.array) => number | mx.array)[],
                              boundaries: number[]) {
  if (schedules.length === 0)
    throw new Error('Must provide at least 1 schedule to join.');
  if (schedules.length !== boundaries.length + 1)
    throw new Error(`Received ${boundaries.length} boundaries but expected ${schedules.length - 1}.`);

  return (step: number | mx.array): mx.array => {
    let output = mx.array(schedules[0](step));
    for (let i = 0; i < boundaries.length; i++) {
      output = mx.where(mx.less(step, boundaries[i]),
                        output,
                        schedules[i + 1](mx.subtract(step, boundaries[i])));
    }
    return output;
  };
}

/**
 * Make a linear scheduler.
 *
 * @remarks
 *
 * @param init Initial value.
 * @param end Final value.
 * @param steps Number of steps to apply the schedule over. The value is
 * ``end`` for any steps beyond ``steps``.
 *
 * @example
 * ```typescript
 * const warmup = optim.linearSchedule(0, 0.1, 100);
 * const optimizer = new optim.Adam(warmup);
 * console.log(optimizer.learningRate); // 0.0
 *
 * for (let i = 0; i < 101; i++) optimizer.update({}, {});
 * console.log(optimizer.learningRate); // 0.1
 * ```
 */
export function linearSchedule(init: number, end: number, steps: number) {
  if (steps < 1)
    throw new Error(`steps must be greater than 0, but got ${steps}.`);

  return (step: number | mx.array) => {
    step = mx.minimum(step, steps);
    return mx.add(mx.multiply(step,
                              mx.divide(mx.subtract(end, init),
                                        steps)),
                  init);
  };
}
