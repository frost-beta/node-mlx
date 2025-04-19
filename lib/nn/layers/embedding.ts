import {core as mx} from '../../core';
import {Module} from './base';
import {QuantizedEmbedding} from './quantized';

/**
 * Implements a simple lookup table that maps each input integer to a
 * high-dimensional vector.
 *
 * Typically used to embed discrete tokens for processing by neural networks.
 */
export class Embedding extends Module {
  weight: mx.array;

  /**
   * Creates an instance of Embedding.
   *
   * @param numEmbeddings - How many possible discrete tokens can we embed.
   * Usually called the vocabulary size.
   * @param dims - The dimensionality of the embeddings.
   */
  constructor(numEmbeddings: number, dims: number) {
    super();
    const scale = Math.sqrt(1 / dims);
    this.weight = mx.random.normal([numEmbeddings, dims], undefined, undefined, scale);
  }

  override toStringExtra(): string {
    return `${this.weight.shape[0]}, ${this.weight.shape[1]}`;
  }

  override forward(x: mx.array): mx.array {
    return this.weight.index(x);
  }

  /**
   * Calls the embedding layer as a linear layer.
   *
   * Use this for example when input embedding and output projection weights are tied.
   *
   * @param x - The input tensor.
   * @returns The output tensor.
   */
  asLinear(x: mx.array): mx.array {
    return mx.matmul(x, this.weight.T);
  }

  /**
   * Return a `QuantizedEmbedding` layer that approximates this embedding layer.
   */
  toQuantized(groupSize = 64, bits = 4): QuantizedEmbedding {
    return QuantizedEmbedding.fromEmbedding(this, groupSize, bits);
  }
}
