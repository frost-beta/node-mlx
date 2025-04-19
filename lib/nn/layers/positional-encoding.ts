import {core as mx} from '../../core';
import {Module} from './base';

/**
 * Implements the rotary positional encoding.
 *
 * @remarks
 *
 * The traditional implementation rotates consecutive pairs of elements in the
 * feature dimension while the default implementation rotates pairs with
 * stride half the feature dimensions for efficiency.
 *
 * For more details see:
 * RoFormer: Enhanced Transformer with Rotary Position Embedding
 * https://arxiv.org/abs/2104.09864
 *
 * @param dims - The feature dimensions to be rotated. If the input feature
 * is larger than dims then the rest is left unchanged.
 * @param traditional - If set to `true` choose the traditional
 * implementation which is slightly less efficient. Default: `false`.
 * @param base - The base used to compute angular frequency for
 * each dimension in the positional encodings. Default: `10000`.
 * @param scale - The scale used to scale the positions. Default: `1.0`.
 */
export class RoPE extends Module {
  dims: number;
  traditional: boolean;
  base: number;
  scale: number;

  constructor(dims: number, traditional = false, base = 10000, scale = 1.0) {
    super();
    this.dims = dims;
    this.traditional = traditional;
    this.base = base;
    this.scale = scale;
  }

  override toStringExtra(): string {
    return `${this.dims}, traditional=${this.traditional}`;
  }

  override forward(x: mx.array, offset = 0) {
    return mx.fast.rope(x, this.dims, this.traditional, this.base, this.scale, offset);
  }
}

/**
 * Implements sinusoidal positional encoding.
 *
 * @remarks
 *
 * For more details see the paper Attention Is All You Need:
 * https://arxiv.org/abs/1706.03762.
 *
 * @param dims - The dimensionality of the resulting positional embeddings.
 * @param minFreq - The minimum frequency expected. Default: `0.0001`.
 * @param maxFreq - The maximum frequency expected. Default: `1`.
 * @param scale - A multiplicative scale for the embeddings.
 * Default: `sqrt(2/dims)`.
 * @param cosFirst - If `true` embed using `[cos(x); sin(x)]`
 * instead of the reverse. Default: `false`.
 * @param fullTurns - If `true` multiply the frequencies with
 * `2 * Math.PI`. Default: `false`.
 */
export class SinusoidalPositionalEncoding extends Module {
  scale: number;
  cosFirst: boolean;

  // Not included in the parameters.
  #sigmas: mx.array;

  constructor(dims: number,
              minFreq = 0.0001,
              maxFreq = 1,
              scale: number | null = null,
              cosFirst = false,
              fullTurns = false) {
    super();
    // Save some constants that define the implementation.
    this.scale = scale ?? Math.sqrt(2 / dims);
    this.cosFirst = cosFirst;

    const oneZero = mx.subtract(1, mx.divide(mx.arange(0, Math.floor(dims / 2)),
                                             dims / 2 - 1));
    minFreq = Math.log(minFreq);
    maxFreq = Math.log(maxFreq);

    this.#sigmas = mx.exp(mx.add(mx.multiply(oneZero,
                                             mx.subtract(maxFreq, minFreq)),
                                 minFreq));
    if (fullTurns) {
      this.#sigmas = mx.multiply(this.#sigmas, 2 * Math.PI);
    }
  }

  override forward(x: mx.array) {
    let y = mx.multiply(x.index('...', null), this.#sigmas);
    const cosy = mx.cos(y);
    const siny = mx.sin(y);

    if (this.cosFirst)
      y = mx.concatenate([cosy, siny], -1);
    else
      y = mx.concatenate([siny, cosy], -1);

    if (this.scale !== 1)
      return mx.multiply(y, this.scale);
    else
      return y;
  }
}

export class ALiBi extends Module {
  static maskKey: (number | mx.Dtype)[] | null = null;
  static mask: mx.array | null = null;

  static createAlibiMatrix(qSequenceLength: number,
                           kSequenceLength: number,
                           numHeads: number,
                           offset: number,
                           dtype = mx.float32): mx.array {
    const maskKey = [qSequenceLength, kSequenceLength, numHeads, offset, dtype];
    if (!ALiBi.maskKey ||
        !maskKey.every((element, i) => element === ALiBi.maskKey[i])) {
      const x1 = mx.arange(offset, qSequenceLength);
      const x2 = mx.arange(0, kSequenceLength);
      const distanceMatrix = mx.negative(mx.abs(mx.expandDims(
        mx.subtract(x1.index(mx.Slice(), null),
                    x2.index(null, mx.Slice())),
        [0, 1])));
      const slope = ALiBi.createAlibiSlope(numHeads);
      const mask = mx.multiply(distanceMatrix, slope).astype(dtype);
      ALiBi.maskKey = maskKey;
      ALiBi.mask = mask;
    }
    return ALiBi.mask;
  }

  static createAlibiSlope(numHeads: number): mx.array {
    const x = Math.pow(Math.pow(2, 8), 1 / numHeads);
    const out = mx.power(x, mx.negative(mx.arange(1, numHeads + 1)));
    return mx.expandDims(out, [-1, -2]);
  }

  forward(attentionScores: mx.array, offset = 0, mask?: mx.array): mx.array {
    let alibiMask = ALiBi.createAlibiMatrix(
      attentionScores.shape[attentionScores.shape.length - 2] + offset,
      attentionScores.shape[attentionScores.shape.length - 1],
      attentionScores.shape[1],
      offset,
      attentionScores.dtype);
    if (mask) {
      alibiMask = mx.add(alibiMask, mask);
    }
    return mx.add(attentionScores, alibiMask);
  }
}
