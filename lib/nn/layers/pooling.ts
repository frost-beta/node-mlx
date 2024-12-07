import {core as mx} from '../../..';
import {accumulate, range} from './pytools';
import {Module} from './base';

class Pool extends Module {
  #poolingFunction: (x: mx.array, axes: number[]) => mx.array;
  #kernelSize: number[];
  #stride: number[];
  #padding: [number, number][];
  #paddingValue: number;
  #axes: number[];

  constructor(poolingFunction: (x: mx.array, axes: number[]) => mx.array,
              kernelSize: number[],
              stride: number[],
              padding: [number, number][],
              paddingValue: number) {
    super();
    this.#poolingFunction = poolingFunction;
    this.#kernelSize = kernelSize;
    this.#stride = stride;
    this.#padding = padding;
    this.#paddingValue = paddingValue;
    this.#axes = range(-kernelSize.length - 1, -1, 1);
  }

  override toStringExtra(): string {
    const pd = this.#padding.map(v => v[0]);
    return `kernelSize=${this.#kernelSize}, stride=${this.#stride}, padding=${pd}`;
  }

  override forward(x: mx.array): mx.array {
    if (this.#padding.some(v => v[0] > 0)) {
      x = mx.pad(x, [[0, 0], ...this.#padding, [0, 0]], this.#paddingValue, 'constant');
    }
    x = slidingWindows(x, this.#kernelSize, this.#stride);
    return this.#poolingFunction(x, this.#axes);
  }
}

class Pool1d extends Pool {
  constructor(poolingFunction: (x: mx.array, axes: number[]) => mx.array,
              paddingValue: number,
              kernelSize: number | [number],
              stride: number | [number] | null = null,
              padding: number | [number] = 0) {
    const msg = '"{}" must be an integer or a tuple containing 1 integer';
    kernelSize = valueOrList(kernelSize, 1, msg.replace('{}', 'kernelSize'));
    if (stride != null)
      stride = valueOrList(stride, 1, msg.replace('{}', 'stride'));
    else
      stride = kernelSize;
    padding = valueOrList(padding, 1, msg.replace('{}', 'padding'))

    super(poolingFunction, kernelSize, stride, padding.map(p => [p, p]), paddingValue);
  }
}

class Pool2d extends Pool {
  constructor(poolingFunction: (x: mx.array, axes: number[]) => mx.array,
              paddingValue: number,
              kernelSize: number | [number, number],
              stride: number | [number, number] | null = null,
              padding: number | [number, number] = 0) {
    const msg = '"{}" must be an integer or a tuple containing 2 integers';
    kernelSize = valueOrList(kernelSize, 2, msg.replace('{}', 'kernelSize'));
    if (stride != null)
      stride = valueOrList(stride, 2, msg.replace('{}', 'stride'));
    else
      stride = kernelSize;
    padding = valueOrList(padding, 2, msg.replace('{}', 'padding'))

    super(poolingFunction, kernelSize, stride, padding.map(p => [p, p]), paddingValue);
  }
}

class Pool3d extends Pool {
  constructor(poolingFunction: (x: mx.array, axes: number[]) => mx.array,
              paddingValue: number,
              kernelSize: number | [number, number, number],
              stride: number | [number, number, number] | null = null,
              padding: number | [number, number, number] = 0) {
    const msg = '"{}" must be an integer or a tuple containing 3 integers';
    kernelSize = valueOrList(kernelSize, 3, msg.replace('{}', 'kernelSize'));
    if (stride != null)
      stride = valueOrList(stride, 3, msg.replace('{}', 'stride'));
    else
      stride = kernelSize;
    padding = valueOrList(padding, 3, msg.replace('{}', 'padding'))

    super(poolingFunction, kernelSize, stride, padding.map(p => [p, p]), paddingValue);
  }
}

/**
 * Applies 1-dimensional max pooling.
 *
 * @remarks
 *
 * Assuming an input of shape `(N, L, C)` and `kernelSize` is `k`, the output is
 * a tensor of shape `(N, L_out, C)`, given by:
 *
 * `out(N_i, t, C_j) = max_{m=0,...,k-1} input(N_i, stride * t + m, C_j)`
 *
 * where `L_out = floor((L + 2 * padding - kernelSize) / stride) + 1`.
 *
 * @param kernelSize - The size of the pooling window kernel.
 * @param stride - The stride of the pooling window. Default: `kernelSize`.
 * @param padding - How much padding to apply to the input. The padding amount
 * is applied to both sides of the spatial axis. Default: `0`.
 */
export class MaxPool1d extends Pool1d {
  constructor(kernelSize: number | [number],
              stride: number | [number] | null = null,
              padding: number | [number] = 0) {
    super(mx.max, -Infinity, kernelSize, stride, padding);
  }
}

/**
 * Applies 1-dimensional average pooling.
 *
 * @remarks
 *
 * Assuming an input of shape `(N, L, C)` and `kernelSize` is `k`, the output is
 * a tensor of shape `(N, L_out, C)`, given by:
 *
 * `out(N_i, t, C_j) = 1/k * sum_{m=0,...,k-1} input(N_i, stride * t + m, C_j)`
 *
 * where `L_out = floor((L + 2 * padding - kernelSize) / stride) + 1`.
 *
 * @param kernelSize - The size of the pooling window kernel.
 * @param stride - The stride of the pooling window. Default: `kernelSize`.
 * @param padding - How much padding to apply to the input. The padding amount
 * is applied to both sides of the spatial axis. Default: `0`.
 */
export class AvgPool1d extends Pool1d {
  constructor(kernelSize: number | [number],
              stride: number | [number] | null = null,
              padding: number | [number] = 0) {
    super(mx.mean, 0, kernelSize, stride, padding);
  }
}

/**
 * Applies 2-dimensional max pooling.
 *
 * @remarks
 *
 * Assuming an input of shape `(N, H, W, C)` and `kernelSize` is `(k_H, k_W)`,
 * the output is a tensor of shape `(N, H_out, W_out, C)`, given by:
 *
 * `out(N_i, h, w, C_j) = max_{m=0,...,k_H-1} max_{n=0,...,k_W-1} input(N_i, stride[0] * h + m, stride[1] * w + n, C_j)`
 *
 * where `H_out = floor((H + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1`
 *       `W_out = floor((W + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1`
 *
 * The parameters `kernelSize`, `stride`, `padding`, can either be:
 *   - a single `number` -- in which case the same value is used for both the
 *     height and width axis;
 *   - a `tuple` of two `numbers`s -- in which case, the first `number` is used
 *     for the height axis, the second `number` for the width axis.
 *
 * @param kernelSize - The size of the pooling window.
 * @param stride - The stride of the pooling window. Default: `kernelSize`.
 * @param padding - How much padding to apply to the input. The padding is
 * applied on both sides of the height and width axis. Default: `0`.
 */
export class MaxPool2d extends Pool2d {
  constructor(kernelSize: number | [number, number],
              stride: number | [number, number] | null = null,
              padding: number | [number, number] = 0) {
    super(mx.max, -Infinity, kernelSize, stride, padding);
  }
}

/**
 * Applies 2-dimensional average pooling.
 *
 * @remarks
 *
 * Assuming an input of shape `(N, H, W, C)` and `kernelSize` is `(kH, kW)`,
 * the output is a tensor of shape `(N, H_out, W_out, C)`, given by:
 *
 * `out(N_i, h, w, C_j) = 1/(kH*kW) * sum_{m=0,...,kH-1} sum_{n=0,...,kW-1}
 *    input(N_i, stride[0] * h + m, stride[1] * w + n, C_j)`
 *
 * where `H_out = floor((H + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1`,
 * `W_out = floor((W + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1`.
 *
 * The parameters `kernelSize`, `stride`, `padding`, can either be:
 *
 * - a single `number` -- in which case the same value is used for both the
 *   height and width axis
 * - a `number[]` of two numbers -- in which case, the first number is used for
 *   the height axis, the second number for the width axis.
 *
 * @param kernelSize - The size of the pooling window.
 * @param stride - The stride of the pooling window. Default: `kernelSize`.
 * @param padding - How much zero padding to apply to the input. The padding is
 * applied on both sides of the height and width axis. Default: `0`.
 */
export class AvgPool2d extends Pool2d {
  constructor(kernelSize: number | [number, number],
              stride: number | [number, number] | null = null,
              padding: number | [number, number] = 0) {
    super(mx.mean, 0, kernelSize, stride, padding);
  }
}

/**
 * Applies 3-dimensional max pooling.
 *
 * @remarks
 *
 * Assuming an input of shape `(N, D, H, W, C)` and `kernelSize` is `(k_D, k_H, k_W)`,
 * the output is a tensor of shape `(N, D_out, H_out, W_out, C)`, given by:
 *
 * `out(N_i, d, h, w, C_j) = max_{l=0,...,k_D-1} max_{m=0,...,k_H-1} max_{n=0,...,k_W-1}
 *                           input(N_i, stride[0] * d + l, stride[1] * h + m, stride[2] * w + n, C_j)`
 *
 * where `D_out = floor((D + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1`
 *       `H_out = floor((H + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1`
 *       `W_out = floor((W + 2 * padding[2] - kernelSize[2]) / stride[2]) + 1`
 *
 * The parameters `kernelSize`, `stride`, `padding`, can either be:
 *   - a single `number` -- in which case the same value is used for the depth,
 *     height and width axis;
 *   - a `tuple` of three `numbers`s -- in which case, the first `number` is used
 *     for the depth axis, the second `number` for the height axis, and the third
 *     `number` for the width axis.
 *
 * @param kernelSize - The size of the pooling window.
 * @param stride - The stride of the pooling window. Default: `kernelSize`.
 * @param padding - How much padding to apply to the input. The padding is
 * applied on both sides of the depth, height and width axis. Default: `0`.
 */
export class MaxPool3d extends Pool3d {
  constructor(kernelSize: number | [number, number, number],
              stride: number | [number, number, number] | null = null,
              padding: number | [number, number, number] = 0) {
    super(mx.max, -Infinity, kernelSize, stride, padding);
  }
}

/**
 * Applies 3-dimensional average pooling.
 *
 * @remarks
 *
 * Assuming an input of shape `(N, D, H, W, C)` and `kernelSize` is `(k_D, k_H, k_W)`,
 * the output is a tensor of shape `(N, D_out, H_out, W_out, C)`, given by:
 *
 * `out(N_i, d, h, w, C_j) = (1 / (k_D * k_H * k_W)) * sum_{l=0,...,k_D-1} sum_{m=0,...,k_H-1} sum_{n=0,...,k_W-1} input(N_i, stride[0] * d + l, stride[1] * h + m, stride[2] * w + n, C_j)`
 *
 * where `D_out = floor((D + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1`
 *       `H_out = floor((H + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1`
 *       `W_out = floor((W + 2 * padding[2] - kernelSize[2]) / stride[2]) + 1`
 *
 * The parameters `kernelSize`, `stride`, `padding`, can either be:
 *
 * - a single `number` -- in which case the same value is used for the depth,
 * height and width axis;
 * - a `tuple` of three `numbers`s -- in which case, the first `number` is used
 * for the depth axis, the second `number` for the height axis, and the third
 * `number` for the width axis.
 *
 * @param kernelSize - The size of the pooling window.
 * @param stride - The stride of the pooling window. Default: `kernelSize`.
 * @param padding - How much zero padding to apply to the input. The padding is
 * applied on both sides of the depth, height and width axis. Default: `0`.
 */
export class AvgPool3d extends Pool3d {
  constructor(kernelSize: number | [number, number, number],
              stride: number | [number, number, number] | null = null,
              padding: number | [number, number, number] = 0) {
    super(mx.mean, 0, kernelSize, stride, padding);
  }
}

function valueOrList<T>(x: number | T, n: number, msg: string): T {
  if (Array.isArray(x)) {
    if (x.length !== n)
      throw Error(msg);
    return x;
  }
  if (typeof x !== 'number')
    throw Error(msg);
  return new Array(n).fill(x) as T;
}

function nonOverlappingSlidingWindows(x: mx.array,
                                      shape: number[],
                                      windowShape: number[]): mx.array {
  // Compute the intermediate shape.
  const newShape = [shape[0]];
  for (let i = 1; i < shape.length - 1; i++) {
    newShape.push(shape[i] / windowShape[i - 1]);
    newShape.push(windowShape[i - 1]);
  }
  newShape.push(shape[shape.length - 1]);

  const lastAxis = newShape.length - 1;
  const axisOrder = [0, ...range(1, lastAxis, 2), ...range(2, lastAxis, 2), lastAxis];

  x = x.reshape(newShape);
  x = x.transpose(axisOrder);
  return x;
}

function slidingWindows(x: mx.array,
                        windowShape: number[],
                        windowStrides: number[]): mx.array {
  if (x.ndim < 3) {
    throw Error(`To extract sliding windows at least 1 spatial dimension ` +
                `(3 total) is needed but the input only has ${x.ndim} dimensions.`);
  }

  const spatialDims = x.shape.slice(1, -1);
  if (spatialDims.length != windowShape.length ||
      spatialDims.length != windowStrides.length) {
    throw Error(`To extract sliding windows the window shapes and strides ` +
                `must have the same number of spatial dimensions as the ` +
                `signal but the signal has ${spatialDims.length} dims and ` +
                `the window shape has ${windowShape.length} and strides have ` +
                `${windowStrides.length}.`);
  }

  const shape = x.shape;
  if (windowShape.every((w, i) => w === windowStrides[i] && spatialDims[i] % w === 0)) {
    return nonOverlappingSlidingWindows(x, shape, windowShape);
  }

  const strides = [...accumulate([...shape, 1].reverse(), (x, y) => x * y)].reverse().slice(1);

  // Compute the output shape.
  const finalShape = [
    shape[0],
    ...windowShape.map((w, i) => (spatialDims[i] - w) / windowStrides[i] + 1),
    ...windowShape,
    shape[shape.length - 1]];

  // Compute the output strides
  const finalStrides = [
    strides[0],
    ...windowStrides.map((s, i) => strides[i + 1] * s),
    ...strides.slice(1, -1),
    strides[strides.length - 1]];

  return mx.asStrided(x, finalShape, finalStrides);
}
