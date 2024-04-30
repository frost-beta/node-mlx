import {core as mx} from '../../..';
import {Module} from './base';

/**
 * A placeholder identity operator that is argument-insensitive.
 *
 * @remarks
 *
 * It does not matter what arguments are passed to the constructor.
 */
export class Identity extends Module {
  constructor(...args: unknown[]) {
    super();
  }

  forward(x: mx.array): mx.array {
    return x;
  }
}

/**
 * Applies an affine transformation to the input.
 *
 * @remarks
 *
 * Concretely:
 *
 * ```math
 * y = x W^\top + b
 * ```
 *
 * where `W` has shape `[output_dims, input_dims]` and `b` has shape `[output_dims]`.
 *
 * The values are initialized from the uniform distribution U(-k, k),
 * where k = 1/sqrt(Di) and Di is equal to `input_dims`.
 *
 * @param inputDims - The dimensionality of the input features.
 * @param outputDims - The dimensionality of the output features.
 * @param bias - If set to `false` then the layer will not use a bias. Default is `true`.
 */
export class Linear extends Module {
  weight: mx.array;
  bias?: mx.array;

  constructor(inputDims: number, outputDims: number, bias = true) {
    super();
    const scale = Math.sqrt(1.0 / inputDims);
    this.weight = mx.random.uniform(-scale, scale, [outputDims, inputDims]);
    if (bias)
      this.bias = mx.random.uniform(-scale, scale, [outputDims]);
  }

  protected toStringExtra(): string {
    return `input_dims=${this.weight.shape[1]}, output_dims=${this.weight.shape[0]}, bias=${'bias' in this}`;
  }

  forward(x: mx.array): mx.array {
    if (this.bias)
      return mx.addmm(this.bias, x, this.weight.T);
    else
      return mx.matmul(x, this.weight.T);
  }
}

/**
 * Applies a bilinear transformation to the inputs.
 *
 * @remarks
 *
 * Concretely:
 *
 * ```math
 * y_i = x_1^\top W_i x_2 + b_i
 * ```
 *
 * where `W` has shape `[output_dims, input1_dims, input2_dims]`, `b` has shape `[output_dims]`,
 * and `i` indexes the output dimension.
 *
 * The values are initialized from the uniform distribution U(-k, k),
 * where `k = 1/sqrt(D1)` and `D1` is `input1_dims`.
 *
 * @param input1Dims - The dimensionality of the input1 features.
 * @param input2Dims - The dimensionality of the input2 features.
 * @param outputDims - The dimensionality of the output features.
 * @param bias - If set to `false` then the layer will not use a bias. Default is `true`.
 */
export class Bilinear extends Module {
  weight: mx.array;
  bias?: mx.array;

  constructor(input1Dims: number, input2Dims: number, outputDims: number, bias = true) {
    super();
    const scale = Math.sqrt(1.0 / input1Dims);
    this.weight = mx.random.uniform(-scale, scale, [outputDims, input2Dims, input1Dims]);
    if (bias)
      this.bias = mx.random.uniform(-scale, scale, [outputDims]);
  }

  protected toStringExtra(): string {
    const [out, in2, in1] = this.weight.shape;
    return `input1_dims=${in1}, input2_dims=${in2}, output_dims=${out}, bias=${'bias' in this}`;
  }

  forward(x1: mx.array, x2: mx.array): mx.array {
    // Normalize shapes.
    const [out, in2, in1] = this.weight.shape;
    const xshape = x1.shape.slice(0, -1);
    x1 = x1.reshape(-1, in1);
    x2 = x2.reshape(-1, 1, in2);

    // Perform the bilinear transformation.
    const w = this.weight.reshape(out * in2, in1);
    let y = mx.matmul(x1, w.T);
    y = y.reshape(-1, out, in2).swapaxes(-2, -1);
    y = mx.matmul(x2, y);
    y = y.squeeze(1);

    // Reset the shape.
    y = y.reshape(...xshape, out);

    // Apply the bias.
    if (this.bias)
      y = mx.add(y, this.bias);

    return y;
  }
}
