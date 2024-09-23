import {core as mx, utils} from '../../..';
import {product} from './pytools';
import {Module} from './base';

/**
 * Upsample the input signal spatially.
 *
 * @remarks
 *
 * The spatial dimensions are by convention dimensions `1` to `x.ndim - 2`. The
 * first is the batch dimension and the last is the feature dimension.
 *
 * For example, an audio signal would be 3D with 1 spatial dimension, an image
 * would be 4D with 2 and so on and so forth.
 *
 * There are three upsampling algorithms implemented: nearest neighbor
 * upsampling, linear interpolation, and cubic interpolation. All can be applied
 * to any number of spatial dimensions. The linear interpolation will be
 * bilinear, trilinear etc when applied to more than one spatial dimension. And
 * cubic interpolation will be bicubic when there are 2 spatial dimensions.
 *
 * When using one of the linear or cubic interpolation modes, the `alignCorners`
 * argument changes how the corners are treated in the input image. If
 * `alignCorners=true` then the top and left edge of the input and output will
 * be matching as will the bottom right edge.
 *
 * @param scaleFactor The multiplier for the spatial size.
 * @param mode The upsampling algorithm, either `"nearest"`, `"linear"` or
 * `"cubic"`. Default: `"nearest"`.
 * @param alignCorners it changes the way the corners are treated during
 * `"linear"` and `"cubic"` upsampling.
 */
export class Upsample extends Module {
  scaleFactor: number | number[];
  mode: string;
  alignCorners: boolean;

  constructor(scaleFactor: number | number[],
              mode: 'nearest' | 'linear' | 'cubic' = 'nearest',
              alignCorners = false) {
    super();
    if (!['nearest', 'linear', 'cubic'].includes(mode)) {
      throw new Error(`[Upsample] Got unsupported upsampling algorithm: ${mode}`);
    }
    this.scaleFactor = scaleFactor;
    this.mode = mode;
    this.alignCorners = alignCorners;
  }

  override toStringExtra(): string {
    return `scaleFactor=${this.scaleFactor}, mode='${this.mode}', alignCorners=${this.alignCorners}`;
  }

  override forward(x: mx.array): mx.array {
    let dims = x.shape.length - 2;
    if (dims <= 0) {
      throw new Error(
        `[Upsample] The input should have at least 1 spatial ` +
        `dimension, which means it should be at least 3 dimensions (D), but ` +
        `${x.shape.length}D was provided`);
    }

    let scaleFactor: number[];
    if (Array.isArray(this.scaleFactor)) {
      if (this.scaleFactor.length !== dims) {
        throw new Error(
          `[Upsample] One scale per spatial dimension is required but `+
          `scaleFactor=${scaleFactor} and the number of spatial dimensions were ${dims}`);
      }
      scaleFactor = this.scaleFactor;
    } else {
      scaleFactor = Array(dims).fill(this.scaleFactor);
    }

    if (this.mode === 'nearest')
      return upsampleNearest(x, scaleFactor);
    else if (this.mode === 'linear')
      return upsampleLinear(x, scaleFactor, this.alignCorners);
    else if (this.mode === 'cubic')
      return upsampleCubic(x, scaleFactor, this.alignCorners);
  }
}

function scaledIndices(N: number,
                       scale: number,
                       alignCorners: boolean,
                       dim: number,
                       ndims: number): mx.array {
  const M = Math.floor(scale * N);
  let indices: mx.array;
  if (alignCorners) {
    indices = mx.multiply(mx.arange(M), (N - 1) / (M - 1));
  } else {
    const step = 1 / scale;
    const start = ((M - 1) * step - N + 1) / 2;
    indices = mx.subtract(mx.multiply(mx.arange(M), step), start);
  }

  const shape = Array(ndims).fill(1);
  shape[dim] = -1;

  return indices.reshape(shape);
}

function nearestIndices(N: number,
                        scale: number,
                        dim: number,
                        ndims: number): mx.array {
  return scaledIndices(N, scale, true, dim, ndims).astype(mx.int32);
}

function linearIndices(N: number,
                       scale: number,
                       alignCorners: boolean,
                       dim: number,
                       ndims: number): [mx.array, mx.array][] {
  let indices = scaledIndices(N, scale, alignCorners, dim, ndims);
  indices = mx.clip(indices, 0, N -1);
  const indicesL = mx.floor(indices);
  const indicesR = mx.ceil(indices);
  const weight = mx.expandDims(mx.subtract(indices, indicesL), -1);

  return [
    [ indicesL.astype(mx.int32), mx.subtract(1, weight) ],
    [ indicesR.astype(mx.int32), weight ],
  ];
}

const getWeight = mx.compile((ind: mx.array, grid: mx.array, dist: number) => {
  const a = -0.75;
  const x = mx.abs(mx.subtract(ind, grid));
  let weight: mx.array;
  if (dist === 1) {
    weight = mx.add(
      mx.multiply(
        mx.multiply(
          mx.subtract(mx.multiply(a + 2.0, x), a + 3.0),
          x),
        x),
      1);
  } else {
    weight = mx.multiply(
      mx.subtract(
        mx.multiply(
          mx.add(mx.multiply(mx.subtract(x, 5), x), 8),
          x),
        4),
      a);
  }
  return weight;
}, true);

function cubicIndices(N: number,
                      scale: number,
                      alignCorners: boolean,
                      dim: number,
                      ndims: number): [mx.array, mx.array][] {
  const indices = scaledIndices(N, scale, alignCorners, dim, ndims);
  let indicesL1 = mx.floor(indices);
  let indicesR1 = mx.floor(mx.add(indices, 1));
  let indicesL2 = mx.subtract(indicesL1, 1);
  let indicesR2 = mx.add(indicesR1, 1);

  const weightL1 = getWeight(indices, indicesL1, 1).index('...', null);
  const weightR1 = getWeight(indices, indicesR1, 1).index('...', null);
  const weightL2 = getWeight(indices, indicesL2, 1).index('...', null);
  const weightR2 = getWeight(indices, indicesR2, 1).index('...', null);

  indicesL1 = mx.clip(indicesL1, 0, N - 1);
  indicesR1 = mx.clip(indicesR1, 0, N - 1);
  indicesL2 = mx.clip(indicesL2, 0, N - 1);
  indicesR2 = mx.clip(indicesR2, 0, N - 1);

  return [
    [ indicesL1.astype(mx.int32), weightL1 ],
    [ indicesR1.astype(mx.int32), weightR1 ],
    [ indicesL2.astype(mx.int32), weightL2 ],
    [ indicesR2.astype(mx.int32), weightR2 ]
  ];
};

export function upsampleNearest(x: mx.array,
                                scaleFactor: number[]): mx.array {
  const dims = x.ndim - 2;
  if (dims !== scaleFactor.length) {
    throw new Error('A scale needs to be provided for each spatial dimension');
  }

  // Integer scaleFactors means we can simply expand-broadcast and reshape.
  if (scaleFactor.every(val => val === Math.floor(val))) {
    let shape = x.shape;
    for (let d = 0; d < dims; d++) {
      shape.splice(2 + 2 * d, 0, 1);
    }
    x = x.reshape(shape);
    for (let d = 0; d < dims; d++) {
      shape[2 + 2 * d] = scaleFactor[d];
    }
    x = mx.broadcastTo(x, shape);
    for (let d = 0; d < dims; d++) {
      shape[d + 1] *= shape[d + 2];
      shape.splice(d + 2, 1);
    }
    x = x.reshape(shape);
    return x;
  } else {
    const [B, ...N] = x.shape;
    const C = N.pop();
    const indices: (mx.array | mx.Slice)[] = [ mx.Slice(null) ];
    for (let i = 0; i < N.length; i++) {
      indices.push(nearestIndices(N[i], scaleFactor[i], i, dims));
    }
    return x.index(...indices);
  }
};

function interpolate(x: mx.array,
                     scaleFactor: number[],
                     indicesFn: (N: number, scale: number, alignCorners: boolean, dim: number, ndims: number) => [mx.array, mx.array][],
                     alignCorners = false): mx.array {
  const dims = x.ndim - 2;
  if (dims !== scaleFactor.length) {
    throw new Error('A scale needs to be provided for each spatial dimension');
  }

  const [B, ...N] = x.shape;
  const C = N.pop();

  // Compute the sampling grid
  let indices = [];
  for (let i = 0; i < N.length; i++) {
    indices.push(indicesFn(N[i], scaleFactor[i], alignCorners, i, dims));
  }

  // Sample and compute the weights.
  let samples = [];
  let weights = [];
  for (const idxWeight of product(...indices)) {
    const idx = [];
    const weight = [];
    for (let i = 0; i < idxWeight.length; i++) {
      idx.push(idxWeight[i][0]);
      weight.push(idxWeight[i][1]);
    }
    samples.push(x.index(mx.Slice(), ...idx));
    weights.push(Array.from(weight).reduce((a: mx.array, b: mx.array) => mx.multiply(a, b)));
  }

  // Interpolate.
  let result = mx.array(0);
  for (let i = 0; i < weights.length; ++i) {
    result = mx.add(result, mx.multiply(weights[i], samples[i]));
  }
  return result;
};

export function upsampleLinear(x: mx.array,
                               scaleFactor: number[],
                               alignCorners = false): mx.array {
  return interpolate(x, scaleFactor, linearIndices, alignCorners);
};

export function upsampleCubic(x: mx.array,
                              scaleFactor: number[],
                              alignCorners = false): mx.array {
  return interpolate(x, scaleFactor, cubicIndices, alignCorners);
};
