import {core as mx} from '../../core';
import {Module} from './base';

/**
 * Applies a 1-dimensional convolution over the multi-channel input sequence.
 *
 * @remarks
 *
 * The channels are expected to be last i.e. the input shape should be `NLC`
 * where:
 *  - `N` is the batch dimension
 *  - `L` is the sequence length
 *  - `C` is the number of input channels
 *
 * @param inChannels - The number of input channels
 * @param outChannels - The number of output channels
 * @param kernelSize - The size of the convolution filters
 * @param stride - The stride when applying the filter. Default: 1.
 * @param padding - How many positions to 0-pad the input with. Default: 0.
 * @param dilation - The dilation of the convolution.
 * @param groups - The number of groups for the convolution. Default: 1.
 * @param bias - If `true` add a learnable bias to the output. Default: `true`
 */
export class Conv1d extends Module {
  stride: number;
  padding: number;
  dilation: number;
  groups: number;
  weight: mx.array;
  bias?: mx.array;

  constructor(inChannels: number,
              outChannels: number,
              kernelSize: number,
              stride = 1,
              padding = 0,
              dilation = 1,
              groups = 1,
              bias = true) {
    super();

    if (inChannels % groups != 0)
      throw new Error(`The number of input channels (${inChannels}) must be divisible by the number of groups (${groups})`);

    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;

    const scale = Math.sqrt(1 / (inChannels * kernelSize));
    this.weight = mx.random.uniform(-scale, scale, [outChannels, kernelSize, inChannels / groups]);
    if (bias) {
      this.bias = mx.zeros([outChannels]);
    }
  }

  override toStringExtra(): string {
    return `${this.weight.shape[2]}, ${this.weight.shape[0]}, ` +
           `kernelSize=${this.weight.shape[1]}, stride=${this.stride}, ` +
           `padding=${this.padding}, dilation=${this.dilation}, ` +
           `groups=${this.groups}, bias=${!!this.bias}`;
  }

  override forward(x: mx.array): mx.array {
    const y = mx.conv1d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
    if (this.bias)
      return mx.add(y, this.bias);
    else
      return y;
  }
}

/**
 * Applies a 2-dimensional convolution over the multi-channel input image.
 *
 * @remarks
 *
 * The channels are expected to be last i.e. the input shape should be `NHWC`
 * where:
 * - `N` is the batch dimension
 * - `H` is the input image height
 * - `W` is the input image width
 * - `C` is the number of input channels
 *
 * @param inChannels - The number of input channels.
 * @param outChannels - The number of output channels.
 * @param kernelSize - The size of the convolution filters.
 * @param stride - The size of the stride when applying the filter. Default: 1.
 * @param padding - How many positions to 0-pad the input with. Default: 0.
 * @param dilation - The dilation of the convolution.
 * @param groups - The number of groups for the convolution. Default: 1.
 * @param bias - If `true` add a learnable bias to the output. Default: `true`
 */
export class Conv2d extends Module {
  stride: number[];
  padding: number[];
  groups: number;
  dilation: number | number[];
  weight: mx.array;
  bias?: mx.array;

  constructor(inChannels: number,
              outChannels: number,
              kernelSize: number | number[],
              stride: number | number[] = [1, 1],
              padding: number | number[] = [0, 0],
              dilation: number | number[] = 1,
              groups = 1,
              bias = true) {
    super();
    this.stride = Array.isArray(stride) ? stride : [stride, stride];
    this.padding = Array.isArray(padding) ? padding : [padding, padding];
    this.dilation = dilation;
    this.groups = groups;

    kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    const scale = Math.sqrt(1 / (inChannels * kernelSize[0] * kernelSize[1]));
    this.weight = mx.random.uniform(-scale, scale, [outChannels, ...kernelSize, inChannels / groups]);
    if (bias) {
      this.bias = mx.zeros([outChannels]);
    }
  }

  override toStringExtra(): string {
    return `${this.weight.shape[3]}, ${this.weight.shape[0]}, ` +
           `kernelSize=${this.weight.shape.slice(1, 3)}, stride=${this.stride}, ` +
           `padding=${this.padding}, dilation=${this.dilation}, ` +
           `groups=${this.groups}, bias=${!!this.bias}`;
  }

  override forward(x: mx.array): mx.array {
    const y = mx.conv2d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
    if (this.bias)
      return mx.add(y, this.bias);
    else
      return y;
  }
}

/**
 * Applies a 3-dimensional convolution over the multi-channel input image.
 *
 * @remarks
 *
 * The channels are expected to be last i.e. the input shape should be `NDHWC`
 * where:
 * - `N` is the batch dimension
 * - `D` is the input image depth
 * - `H` is the input image height
 * - `W` is the input image width
 * - `C` is the number of input channels
 *
 * @param inChannels - The number of input channels.
 * @param outChannels - The number of output channels.
 * @param kernelSize - The size of the convolution filters.
 * @param stride - The size of the stride when applying the filter. Default: 1.
 * @param padding - How many positions to 0-pad the input with. Default: 0.
 * @param dilation - The dilation of the convolution.
 * @param bias - If `true` add a learnable bias to the output. Default: `true`
 */
export class Conv3d extends Module {
  stride: number[];
  padding: number[];
  dilation: number | number[];
  weight: mx.array;
  bias?: mx.array;

  constructor(inChannels: number,
              outChannels: number,
              kernelSize: number | number[],
              stride: number | number[] = [1, 1, 1],
              padding: number | number[] = [0, 0, 0],
              dilation: number | number[] = [1, 1, 1],
              bias = true) {
    super();
    this.stride = Array.isArray(stride) ? stride : [stride, stride, stride];
    this.padding = Array.isArray(padding) ? padding : [padding, padding, padding];
    this.dilation = dilation;

    kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    const scale = Math.sqrt(1 / (inChannels * kernelSize[0] * kernelSize[1] * kernelSize[2]));
    this.weight = mx.random.uniform(-scale, scale, [outChannels, ...kernelSize, inChannels]);
    if (bias) {
      this.bias = mx.zeros([outChannels]);
    }
  }

  override toStringExtra(): string {
    return `${this.weight.shape[3]}, ${this.weight.shape[0]}, ` +
           `kernelSize=${this.weight.shape.slice(1, 4)}, stride=${this.stride}, ` +
           `padding=${this.padding}, dilation=${this.dilation}, ` +
           `bias=${!!this.bias}`;
  }

  override forward(x: mx.array): mx.array {
    const y = mx.conv3d(x, this.weight, this.stride, this.padding, this.dilation);
    if (this.bias)
      return mx.add(y, this.bias);
    else
      return y;
  }
}
