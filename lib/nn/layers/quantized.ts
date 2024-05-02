import {core as mx, utils} from '../../..';
import {Embedding} from './embedding';
import {Linear} from './linear';
import {Module} from './base';

/**
 * Quantize the sub-modules of a module according to a predicate.
 *
 * @remarks
 *
 * By default all `Linear` and `Embedding` layers will be quantized. Note also,
 * the module is updated in-place.
 *
 * @param model - The model whose leaf modules may be quantized.
 * @param groupSize - The quantization group size. Default: `64`.
 * @param bits - The number of bits per parameter. Default: `4`.
 * @param classPredicate - A function which receives the `Module` path and
 * `Module` itself and returns `true` if it should be quantized and `false`
 * otherwise. If `None`, then all linear and embedding layers are quantized.
 * Default: `None`.
 */
export function quantize(model: Module,
                         groupSize = 64,
                         bits = 4,
                         classPredicate = (p, m) => m instanceof Linear || m instanceof Embedding): void {
  function maybeQuantize(path: string, m: Module): Module {
    if (!classPredicate(path, m))
      return m;
    if (m instanceof Linear)
      return QuantizedLinear.fromLinear(m, groupSize, bits);
    if (m instanceof Embedding)
      return QuantizedEmbedding.fromEmbedding(m, groupSize, bits);
    throw Error(`Unable to quantize model of type ${typeof m}`);
  }

  const leaves = model.leafModules();
  const leavesWithPaths = utils.treeMapWithPath(maybeQuantize, leaves, undefined, Module.isModule);
  model.updateModules(leavesWithPaths as {[key: string]: Module});
}

/**
 * The same as `Embedding` but with a quantized weight matrix.
 *
 * @remarks
 *
 * `QuantizedEmbedding` also provides a `fromEmbedding` class method to convert
 * embedding layers to `QuantizedEmbedding` layers.
 */
export class QuantizedEmbedding extends Module {
  /**
   * Create a `QuantizedEmbedding` layer from an `Embedding` layer.
   */
  static fromEmbedding(embeddingLayer: Embedding, groupSize = 64, bits = 4): QuantizedEmbedding {
    const embeddingDims = embeddingLayer.weight.shape[0];
    const dims = embeddingLayer.weight.shape[1];
    const instance = new QuantizedEmbedding(embeddingDims, dims, groupSize, bits);
    [instance.weight, instance.scales, instance.biases] = mx.quantize(embeddingLayer.weight, groupSize, bits);
    return instance;
  }

  groupSize: number;
  bits: number;
  numEmbeddings: number;
  dims: number;
  weight: mx.array;
  scales: mx.array;
  biases: mx.array;

  /**
   * Construct a new `QuantizedEmbedding` instance.
   *
   * @param numEmbeddings - How many possible discrete tokens can we embed.
   * Usually called the vocabulary size.
   * @param dims - The dimensionality of the embeddings.
   * @param groupSize - The group size to use for the quantized weight.
   * Default: `64`.
   * @param bits - The bit width to use for the quantized weight. Default: `4`.
   */
  constructor(numEmbeddings: number, dims: number, groupSize = 64, bits = 4) {
    super();

    // Quantization config.
    this.groupSize = groupSize;
    this.bits = bits;

    // Initialize the quantized weight.
    const scale = Math.sqrt(1 / dims);
    const weight = mx.random.normal([numEmbeddings, dims], undefined, undefined, scale);
    [this.weight, this.scales, this.biases] = mx.quantize(weight, groupSize, bits);
    this.numEmbeddings = numEmbeddings;
    this.dims = dims;

    // Freeze this model's parameters.
    this.freeze();
  }

  override forward(x: mx.array): mx.array {
    const s = x.shape;
    x = x.flatten();
    const out = mx.dequantize(this.weight.index(x),
                              this.scales.index(x),
                              this.biases.index(x),
                              this.groupSize,
                              this.bits);
    return out.reshape(...s, -1);
  }

  /**
   * Call the quantized embedding layer as a quantized linear layer.
   *
   * @remarks
   *
   * Use this for example when input embedding and output projection weights are tied.
   */
  asLinear(x: mx.array): mx.array {
    return mx.quantizedMatmul(x,
                              this.weight,
                              this.scales,
                              this.biases,
                              true,
                              this.groupSize,
                              this.bits);
  }

  override toStringExtra(): string {
    return `${this.numEmbeddings}, ${this.dims}, groupSize=${this.groupSize}, bits=${this.bits}`;
  }
}

/**
 * Applies an affine transformation to the input using a quantized weight matrix.
 *
 * @remarks
 *
 * It is the quantized equivalent of `Linear`. For now its parameters are frozen
 * and will not be included in any gradient computation but this will probably change
 * in the future.
 *
 * `QuantizedLinear` also provides a classmethod `fromLinear` to convert
 * linear layers to `QuantizedLinear` layers.
 */
export class QuantizedLinear extends Module {
  /**
   * Create a `QuantizedLinear` layer from a `Linear` layer.
   */
  static fromLinear(linearLayer: Linear, groupSize = 64, bits = 4): QuantizedLinear {
    const [outDims, inDims] = linearLayer.weight.shape;
    const ql = new QuantizedLinear(inDims, outDims, false, groupSize, bits);
    [ql.weight, ql.scales, ql.biases] = mx.quantize(linearLayer.weight, groupSize, bits);
    if (linearLayer.bias)
      ql.bias = linearLayer.bias;
    return ql;
  }

  groupSize: number;
  bits: number;
  weight: mx.array;
  scales: mx.array;
  biases: mx.array;
  bias?: mx.array;

  /**
   * Construct a new `QuantizedLinear` instance.
   *
   * @param inDims - The dimensionality of the input features.
   * @param outDims - The dimensionality of the output features.
   * @param bias - If set to `false` then the layer will not use a bias.
   *   Default: `true`.
   * @param groupSize - The group size to use for the quantized weight.
   *   Default: `64`.
   * @param bits - The bit width to use for the quantized weight. Default: `4`.
   */
  constructor(inDims: number, outDims: number, bias = true, groupSize = 64, bits = 4) {
    super();

    // Quantization config.
    this.groupSize = groupSize;
    this.bits = bits;

    // Initialize the quantized weight.
    const scale = Math.sqrt(1 / inDims);
    const weight = mx.random.uniform(-scale, scale, [outDims, inDims]);
    [this.weight, this.scales, this.biases] = mx.quantize(weight, groupSize, bits);

    // And bias if needed.
    if (bias)
      this.bias = mx.zeros([outDims]);

    // Freeze this model's parameters.
    this.freeze();
  }

  override toStringExtra(): string {
    let [outDims, inDims] = this.weight.shape;
    inDims = inDims * 32 / this.bits;
    return `inputDims=${inDims}, outputDims=${outDims}, bias=${!!this.bias}, `+
           `groupSize=${this.groupSize}, bits=${this.bits}`;
  }

  override forward(x: mx.array): mx.array {
    x = mx.quantizedMatmul(x, this.weight, this.scales, this.biases, true, this.groupSize, this.bits);
    if (this.bias)
      x = mx.add(x, this.bias);
    return x;
  }

  // Wrap unfreeze so that we unfreeze any layers we might contain but
  // our parameters will remain frozen.
  override unfreeze(...args): this {
    super.unfreeze(...args);
    return this.freeze(false);
  }
}
