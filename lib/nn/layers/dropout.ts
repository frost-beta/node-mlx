import {core as mx, utils} from '../../..';
import {Module} from './base';

/**
 * Randomly zero a portion of the elements during training.
 *
 * @remarks
 *
 * The remaining elements are multiplied with `\frac{1}{1-p}` where `p` is the
 * probability of zeroing an element. This is done so the expected value of a
 * given element will remain the same.
 *
 * @param p - The probability to zero an element
 */
export class Dropout extends Module {
  #p1: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p >= 1) {
      throw Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this.#p1 = 1 - p;
  }

  override toStringExtra(): string {
    return `p=${1 - this.#p1}`;
  }

  override forward(x: mx.array): mx.array {
    if (this.#p1 === 1 || !this.training) {
      return x;
    }
    const mask = mx.random.bernoulli(this.#p1, x.shape);
    return mx.multiply(mx.multiply(mask, x), mx.array(1 / this.#p1, x.dtype));
  }
}

/**
 * Apply 2D channel-wise dropout during training.
 *
 * @remarks
 *
 * Randomly zero out entire channels independently with probability `p`. This
 * layer expects the channels to be last, i.e. the input shape should be `NWHC`
 * or `WHC` where `N` is the batch dimension, `H` is the input image height, `W`
 * is the input image width, and `C` is the number of input channels.
 *
 * The remaining channels are scaled by `\frac{1}{1-p}` to maintain the expected
 * value of each element. Unlike traditional dropout, which zeros individual
 * entries, this layer zeros entire channels. This is beneficial for early
 * convolution layers where adjacent pixels are correlated. In such case,
 * traditional dropout may not effectively regularize activations.
 *
 * For more details, see: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and
 * Bregler C., 2015. Efficient Object Localization Using Convolutional Networks.
 * CVPR 2015.
 *
 * @param p - Probability of zeroing a channel during training.
 */
export class Dropout2d extends Module {
  #p1: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p >= 1) {
      throw Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this.#p1 = 1 - p;
  }

  override toStringExtra(): string {
    return `p=${1 - this.#p1}`;
  }

  override forward(x: mx.array): mx.array {
    if (!(x.ndim === 3 || x.ndim === 4)) {
      throw Error(`Received input with ${x.ndim} dimensions. Expected 3 or 4 dimensions.`);
    }

    if (this.#p1 === 1 || !this.training) {
      return x;
    }

    // Dropout is applied on the whole channel.
    // 3D input: (1, 1, C)
    // 4D input: (B, 1, 1, C)
    const maskShape = [...x.shape];
    maskShape[maskShape.length - 2] = 1;
    maskShape[maskShape.length - 3] = 1;

    const mask = mx.random.bernoulli(this.#p1, maskShape);
    return mx.multiply(mx.multiply(mask, x), mx.array(1 / this.#p1, x.dtype));
  }
}

/**
 * Apply 3D channel-wise dropout during training.
 *
 * @remarks
 *
 * Randomly zero out entire channels independently with probability `p`. This
 * layer expects the channels to be last, i.e., the input shape should be
 * `NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth, `H` is
 * the input image height, `W` is the input image width, and `C` is the number
 * of input channels.
 *
 * The remaining channels are scaled by `\frac{1}{1-p}` to maintain the expected
 * value of each element. Unlike traditional dropout, which zeros individual
 * entries, this layer zeros entire channels. This is often beneficial for
 * convolutional layers processing 3D data, like in medical imaging or video
 * processing.
 *
 * @param p - Probability of zeroing a channel during training.
 */
export class Dropout3d extends Module {
  #p1: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p >= 1) {
      throw Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this.#p1 = 1 - p;
  }

  override toStringExtra(): string {
    return `p=${1 - this.#p1}`;
  }

  override forward(x: mx.array): mx.array {
    if (x.ndim != 4 && x.ndim != 5) {
      throw Error(`Received input with ${x.ndim} dimensions. Expected 4 or 5 dimensions.`);
    }

    if (this.#p1 === 1 || !this.training) {
      return x;
    }

    const maskShape = [...x.shape];
    maskShape[maskShape.length - 2] = 1;
    maskShape[maskShape.length - 3] = 1;
    maskShape[maskShape.length - 4] = 1;

    const mask = mx.random.bernoulli(this.#p1, maskShape);
    return mx.multiply(mx.multiply(mask, x), mx.array(1 / this.#p1, x.dtype));
  }
}
