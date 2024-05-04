import {core as mx} from '../../..';
import {checkpoint} from '../utils';
import {relu} from './activations';
import {Module} from './base';
import {Dropout} from './dropout';
import {Linear} from './linear';
import {LayerNorm} from './normalization';

/**
 * Implements the scaled dot product attention with multiple heads.
 *
 * @remarks
 *
 * Given inputs for queries, keys and values, the MultiHeadAttention produces
 * new values by aggregating information from the input values according to the
 * similarities of the input queries and keys.
 *
 * All inputs as well as the output are linearly projected, without biases by
 * default.
 *
 * MultiHeadAttention also takes an optional additive attention mask that should
 * be broadcastable with (batch, numHeads, # queries, # keys). The mask should
 * have `-Infinity` or very large negative numbers at the positions that should
 * *not* be attended to.
 *
 * @param dims The model dimensions. This is also the default value for the
 * queries, keys, values, and the output.
 * @param numHeads The number of attention heads to use.
 * @param queryInputDims The input dimensions of the queries. Default: `dims`.
 * @param keyInputDims The input dimensions of the keys. Default: `dims`.
 * @param valueInputDims The input dimensions of the values. Default:
 * `keyInputDims`.
 * @param valueDims The dimensions of the values after the projection. Default:
 * `dims`.
 * @param valueOutputDims The dimensions the new values will be projected to.
 * Default: `dims`.
 * @param bias Whether or not to use a bias in the projections. Default:
 * `false`.
 */
export class MultiHeadAttention extends Module {
  numHeads: number;
  queryProj: Linear;
  keyProj: Linear;
  valueProj: Linear;
  outProj: Linear;

  constructor(dims: number,
              numHeads: number,
              queryInputDims: number | null = null,
              keyInputDims: number | null = null,
              valueInputDims: number | null = null,
              valueDims: number | null = null,
              valueOutputDims: number | null = null,
              bias: boolean = false) {
    if (dims % numHeads !== 0) {
      throw new Error(`The input feature dimensions should be divisible by the ` +
                      `number of heads (${dims} % ${numHeads}) != 0`);
    }
    super();

    queryInputDims = queryInputDims ?? dims;
    keyInputDims = keyInputDims ?? dims;
    valueInputDims = valueInputDims ?? keyInputDims;
    valueDims = valueDims ?? dims;
    valueOutputDims = valueOutputDims ?? dims;

    this.numHeads = numHeads;
    this.queryProj = new Linear(queryInputDims, dims, bias);
    this.keyProj = new Linear(keyInputDims, dims, bias);
    this.valueProj = new Linear(valueInputDims, valueDims, bias);
    this.outProj = new Linear(valueDims, valueOutputDims, bias);
  }

  override forward(queries: mx.array,
                   keys: mx.array,
                   values: mx.array,
                   mask: mx.array | null = null) {
    queries = this.queryProj.forward(queries);
    keys = this.keyProj.forward(keys);
    values = this.valueProj.forward(values);

    const numHeads = this.numHeads;
    const [B, L, D] = queries.shape;
    const S = keys.shape[1];
    queries = queries.reshape(B, L, numHeads, -1).transpose(0, 2, 1, 3);
    keys = keys.reshape(B, S, numHeads, -1).transpose(0, 2, 3, 1);
    values = values.reshape(B, S, numHeads, -1).transpose(0, 2, 1, 3);

    // Dimensions are [batch x numHeads x sequence x hiddenDim].
    const scale = mx.array(Math.sqrt(1 / queries.shape[-1]), queries.dtype);
    let scores = mx.matmul(mx.multiply(queries, scale), keys);
    if (mask)
      scores = mx.add(scores, mask.astype(scores.dtype));
    scores = mx.softmax(scores, -1)
    const valuesHat = mx.matmul(scores, values).transpose(0, 2, 1, 3)
                                               .reshape(B, L, -1);

    return this.outProj.forward(valuesHat);
  }

  /**
   * Creates an additive causal mask.
   *
   * @param N The size of mask.
   * @param dtype The data type of mask.
   * @returns The mask of shape [N, N].
   */
  static createAdditiveCausalMask(N: number,
                                  dtype: mx.Dtype = mx.float32): mx.array {
    const indices = mx.arange(N, mx.int32);
    let mask = mx.less(indices.index(mx.Slice(), null),
                       indices.index(null));
    // Usually `infinity` but 1e9 is as good and softmax(full(1e9)) != NaN.
    mask = mx.multiply(mask.astype(dtype), -1e9);
    return mask;
  }
}

export class TransformerEncoderLayer extends Module {
  mlpDims: number;
  attention: MultiHeadAttention;
  ln1: LayerNorm;
  ln2: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  dropout1: Dropout;
  dropout2: Dropout;
  activation: (x: mx.array) => mx.array;
  normFirst: boolean;

  constructor(dims: number,
              numHeads: number,
              mlpDims: number | null = null,
              dropout: number = 0,
              activation: (x: mx.array) => mx.array = relu,
              normFirst: boolean = true) {
    super();

    mlpDims = mlpDims || dims * 4;
    this.attention = new MultiHeadAttention(dims, numHeads);
    this.ln1 = new LayerNorm(dims);
    this.ln2 = new LayerNorm(dims);
    this.linear1 = new Linear(dims, mlpDims);
    this.linear2 = new Linear(mlpDims, dims);
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.activation = activation;
    this.normFirst = normFirst;
  }

  override forward(x: mx.array, mask: mx.array) {
    let y: mx.array;
    if (this.normFirst) {
      y = this.ln1.forward(x);
      y = this.attention.forward(y, y, y, mask);
      y = this.dropout1.forward(y);
      x = mx.add(x, y);

      y = this.ln2.forward(x);
      y = this.linear1.forward(y);
      y = this.activation(y);
      y = this.dropout2.forward(y);
      y = this.linear2.forward(y);
      y = mx.add(x, y);
    } else {
      y = this.attention.forward(x, x, x, mask);
      y = this.dropout1.forward(y);
      y = this.ln1.forward(mx.add(x, y));

      y = this.linear1.forward(y);
      y = this.activation(y);
      y = this.dropout2.forward(y);
      y = this.linear2.forward(y);
      y = this.ln2.forward(mx.add(x, y));
    }

    return y;
  }
}

export class TransformerEncoder extends Module {
  layers: TransformerEncoderLayer[];
  ln: LayerNorm;
  checkpoint: boolean;

  constructor(numLayers: number,
              dims: number,
              numHeads: number,
              mlpDims: number | null = null,
              dropout: number = 0,
              activation: (x: mx.array) => mx.array = relu,
              normFirst: boolean = true,
              checkpoint: boolean = false) {
    super();

    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      const layer = new TransformerEncoderLayer(
          dims, numHeads, mlpDims, dropout, activation, normFirst);
      this.layers.push(layer);
    }
    this.ln = new LayerNorm(dims);
    this.checkpoint = checkpoint;
  }

  override forward(x: mx.array, mask: mx.array) {
    for (const layer of this.layers) {
      if (this.checkpoint)
        x = checkpoint(layer)(x, mask);
      else
        x = layer.forward(x, mask);
    }
    return this.ln.forward(x);
  }
}

export class TransformerDecoderLayer extends Module {
  selfAttention: MultiHeadAttention;
  crossAttention: MultiHeadAttention;
  ln1: LayerNorm;
  ln2: LayerNorm;
  ln3: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  dropout1: Dropout;
  dropout2: Dropout;
  dropout3: Dropout;
  activation: (x: mx.array) => mx.array;
  normFirst: boolean;

  constructor(dims: number,
              numHeads: number,
              mlpDims: number | null = null,
              dropout: number = 0,
              activation: (x: mx.array) => mx.array = relu,
              normFirst: boolean = true) {
    super();

    mlpDims = mlpDims ?? dims * 4;
    this.selfAttention = new MultiHeadAttention(dims, numHeads);
    this.crossAttention = new MultiHeadAttention(dims, numHeads);
    this.ln1 = new LayerNorm(dims);
    this.ln2 = new LayerNorm(dims);
    this.ln3 = new LayerNorm(dims);
    this.linear1 = new Linear(dims, mlpDims);
    this.linear2 = new Linear(mlpDims, dims);
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropout3 = new Dropout(dropout);
    this.activation = activation;
    this.normFirst = normFirst;
  }

  override forward(x: mx.array,
                   memory: mx.array,
                   xMask: mx.array,
                   memoryMask: mx.array | null) {
    let y: mx.array;
    if (this.normFirst) {
      y = this.ln1.forward(x);
      y = this.selfAttention.forward(y, y, y, xMask);
      y = this.dropout1.forward(y);
      x = mx.add(x, y);

      y = this.ln2.forward(x);
      y = this.crossAttention.forward(y, memory, memory, memoryMask);
      y = this.dropout2.forward(y);
      x = mx.add(x, y);

      y = this.ln3.forward(x);
      y = this.linear1.forward(y);
      y = this.activation(y);
      y = this.dropout3.forward(y);
      y = this.linear2.forward(y);
      y = mx.add(x, y);
    } else {
      y = this.selfAttention.forward(x, x, x, xMask);
      y = this.dropout1.forward(y);
      x = this.ln1.forward(mx.add(x, y));

      y = this.crossAttention.forward(x, memory, memory, memoryMask);
      y = this.dropout2.forward(y);
      x = this.ln2.forward(mx.add(x, y));

      y = this.linear1.forward(x);
      y = this.activation(y);
      y = this.dropout3.forward(y);
      y = this.linear2.forward(y);
      y = this.ln3.forward(mx.add(x, y));
    }

    return y;
  }
}

export class TransformerDecoder extends Module {
  layers: TransformerDecoderLayer[];
  ln: LayerNorm;
  checkpoint: boolean;

  constructor(numLayers: number,
              dims: number,
              numHeads: number,
              mlpDims: number | null = null,
              dropout: number = 0,
              activation: (x: mx.array) => mx.array = relu,
              normFirst: boolean = true,
              checkpoint: boolean = false) {
    super();

    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      const layer = new TransformerDecoderLayer(
          dims, numHeads, mlpDims, dropout, activation, normFirst);
      this.layers.push(layer);
    }
    this.ln = new LayerNorm(dims);
    this.checkpoint = checkpoint;
  }

  override forward(x: mx.array,
                   memory: mx.array,
                   xMask: mx.array,
                   memoryMask: mx.array | null) {
    for (let layer of this.layers) {
      if (this.checkpoint)
        x = checkpoint(layer)(x, memory, xMask, memoryMask);
      else
        x = layer.forward(x, memory, xMask, memoryMask);
    }
    return this.ln.forward(x);
  }
}

/**
 * Implements a standard Transformer model.
 *
 * @remarks
 *
 * The implementation is based on [Attention Is All You
 * Need](https://arxiv.org/abs/1706.03762).
 *
 * The Transformer model contains an encoder and a decoder. The encoder
 * processes the input sequence and the decoder generates the output sequence.
 * The interaction between encoder and decoder happens through the attention
 * mechanism.
 *
 * @param dims The number of expected features in the encoder/decoder inputs.
 * Default: `512`.
 * @param numHeads The number of attention heads. Default: `8`.
 * @param numEncoderLayers The number of encoder layers in the Transformer
 * encoder. Default: `6`.
 * @param numDecoderLayers The number of decoder layers in the Transformer
 * decoder. Default: `6`.
 * @param mlpDims The hidden dimension of the MLP block in each Transformer
 * layer. Defaults to `4 * dims` if not provided. Default: `null`.
 * @param dropout The dropout value for the Transformer encoder and decoder.
 * Dropout is used after each attention layer and the activation in the MLP
 * layer. Default: `0.0`.
 * @param activation The activation function for the MLP hidden layer.
 * Default: `mx.relu`.
 * @param customEncoder A custom encoder to replace the standard Transformer
 * encoder. Default: `null`.
 * @param customDecoder A custom decoder to replace the standard Transformer
 * decoder. Default: `null`.
 * @param normFirst If `true`, encoder and decoder layers will perform layer
 * normalization before attention and MLP operations, otherwise after.
 * Default: `true`.
 * @param checkpoint If `true` perform gradient checkpointing to reduce the
 * memory usage at the expense of more computation. Default: `false`.
 */
export class Transformer extends Module {
  encoder: TransformerEncoder;
  decoder: TransformerDecoder;

  constructor(dims: number = 512,
              numHeads: number = 8,
              numEncoderLayers: number = 6,
              numDecoderLayers: number = 6,
              mlpDims: number | null = null,
              dropout: number = 0,
              activation: (x: mx.array) => mx.array = relu,
              customEncoder: TransformerEncoder | null = null,
              customDecoder: TransformerDecoder | null = null,
              normFirst: boolean = true,
              checkpoint: boolean = false) {
    super();

    this.encoder = customEncoder || new TransformerEncoder(
      numEncoderLayers,
      dims,
      numHeads,
      mlpDims,
      dropout,
      activation,
      normFirst,
      checkpoint
    );

    this.decoder = customDecoder || new TransformerDecoder(
      numDecoderLayers,
      dims,
      numHeads,
      mlpDims,
      dropout,
      activation,
      normFirst,
      checkpoint
    );
  }

  override forward(src: mx.array,
                   tgt: mx.array,
                   srcMask: mx.array,
                   tgtMask: mx.array,
                   memoryMask: mx.array) {
    const memory = this.encoder.forward(src, srcMask);
    return this.decoder.forward(tgt, memory, tgtMask, memoryMask);
  }
}