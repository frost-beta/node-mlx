import {core as mx, utils} from '../../..';
import {range} from './pytools';
import {Module} from './base';

/**
 * Applies instance normalization on the inputs.
 *
 * @remarks
 *
 * Computes
 *
 * ```math
 * y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta,
 * ```
 *
 * where `gamma` and `beta` are learned per feature dimension parameters
 * initialized at 1 and 0 respectively. Both are of size `dims`, if `affine` is
 * `true`.
 *
 * Reference: https://arxiv.org/abs/1607.08022
 *
 * @param dims - The number of features of the input.
 * @param eps - A value added to the denominator for numerical stability.
 * Default: `1e-5`.
 * @param affine - Default: `false`.
 *
 * @example
 * ```typescript
 * import {core as mx} from '@frost-beta/mlx';
 *
 * const x = mx.random.normal([8, 4, 4, 16]);
 * const inorm = new nn.InstanceNorm(16);
 * const output = inorm.forward(x);
 * ```
 */
export class InstanceNorm extends Module {
  dims: number;
  eps: number;
  weight?: mx.array;
  bias?: mx.array;

  constructor(dims: number, eps = 1e-5, affine = false) {
    super();
    this.dims = dims;
    this.eps = eps;
    if (affine) {
      this.weight = mx.ones([dims]);
      this.bias = mx.ones([dims]);
    }
  }

  override toStringExtra(): string {
    return `${this.dims}, eps=${this.eps.toExponential()}, affine=${!!this.weight}`;
  }

  override forward(x: mx.array): mx.array {
    const reductionAxes = range(1, x.ndim - 1);
    // Compute stats.
    const mean = mx.mean(x, reductionAxes, true);
    const variance = mx.variance(x, reductionAxes, true);
    // Normalize.
    x = mx.multiply(mx.subtract(x, mean),
                    mx.rsqrt(mx.add(variance, this.eps)));
    // Scale and shift if necessary.
    if (this.weight)
      return mx.add(mx.multiply(this.weight, x), this.bias);
    else
      return x;
  }
}

/**
 * Applies layer normalization on the inputs.
 *
 * @remarks
 *
 * Computes
 *
 * ```math
 * y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,
 * ```
 *
 * where `gamma` and `beta` are learned per feature dimension parameters
 * initialized at 1 and 0 respectively.
 *
 * Reference: https://arxiv.org/abs/1607.06450
 *
 * @param dims - The feature dimension of the input to normalize over.
 * @param eps - A small additive constant for numerical stability.
 * @param affine - If true learn an affine transform to apply after the
 * normalization.
 * @param bias - If true include a translation to the affine transformation. If
 * it's false the transformation is not affine but just scaling.
 */
export class LayerNorm extends Module {
  dims: number;
  eps: number;
  weight?: mx.array;
  bias?: mx.array;

  constructor(dims: number, eps = 1e-5, affine = true, bias = true) {
    super();
    this.dims = dims;
    this.eps = eps;
    if (affine) {
      this.weight = mx.ones([dims]);
      if (bias) {
        this.bias = mx.ones([dims]);
      }
    }
  }

  override toStringExtra(): string {
    return `${this.dims}, eps=${this.eps.toExponential()}, affine=${!!this.weight}`;
  }

  override forward(x: mx.array): mx.array {
    return mx.fast.layerNorm(x, this.weight, this.bias, this.eps);
  }
}

/**
 * Applies Root Mean Square normalization to the inputs.
 *
 * @remarks
 *
 * Computes
 *
 * ```math
 * y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma
 * ```
 *
 * where `gamma` is a learned per feature dimension parameter initialized at 1.
 *
 * Reference: https://arxiv.org/abs/1910.07467
 *
 * @param dims - The feature dimension of the input to normalize over.
 * @param eps - A small additive constant for numerical stability.
 */
export class RMSNorm extends Module {
  weight: mx.array;
  eps: number;

  constructor(dims: number, eps = 1e-5) {
    super();
    this.weight = mx.ones([dims]);
    this.eps = eps;
  }

  override toStringExtra(): string {
    return `${this.weight.shape[0]}, eps=${this.eps.toExponential()}`;
  }

  override forward(x: mx.array): mx.array {
    return mx.fast.rmsNorm(x, this.weight, this.eps);
  }
}

/**
 * Applies Group Normalization [1] to the inputs.
 *
 * @remarks
 *
 * Computes the same normalization as layer norm, namely
 *
 * ```math
 * y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,
 * ```
 *
 * where `gamma` and `beta` are learned per feature dimension parameters
 * initialized at 1 and 0 respectively.
 *
 * The mean and variance are computed over the spatial dimensions and each group
 * of features. In particular, the input is split into numGroups across the
 * feature dimension.
 *
 * The feature dimension is assumed to be the last dimension and the dimensions
 * that precede it (except the first) are considered the spatial dimensions.
 *
 * Reference: https://arxiv.org/abs/1803.08494
 *
 * @param numGroups - Number of groups to separate the features into.
 * @param dims - The feature dimensions of the input to normalize over.
 * @param eps - A small additive constant for numerical stability.
 * @param affine - If `true` learn an affine transform to apply after the
 * normalization.
 * @param pyTorchCompatible - If `true` perform the group normalization in the
 * same order/grouping as PyTorch.
 */
export class GroupNorm extends Module {
  numGroups: number;
  dims: number;
  eps: number;
  pyTorchCompatible: boolean;
  bias?: mx.array;
  weight?: mx.array;

  constructor(numGroups: number, dims: number, eps = 1e-5, affine = true, pyTorchCompatible = false) {
    super();
    this.numGroups = numGroups;
    this.dims = dims;
    this.eps = eps;
    this.pyTorchCompatible = pyTorchCompatible;
    if (affine) {
      this.bias = mx.zeros([dims]);
      this.weight = mx.ones([dims]);
    }
  }

  override toStringExtra() {
    return `${this.numGroups}, ${this.dims}, eps=${this.eps.toExponential()}, affine=${!!this.weight}, pyTorchCompatible=${this.pyTorchCompatible}`;
  }

  override forward(x: mx.array) {
    if (this.pyTorchCompatible)
      x = this.pyTorchCompatibleGroupNorm(x);
    else
      x = this.groupNorm(x);
    if (this.weight)
      return mx.add(mx.multiply(this.weight, x), this.bias);
    else
      return x;
  }

  private pyTorchCompatibleGroupNorm(x: mx.array) {
    const [dims] = x.shape.slice(-1);
    const [batch, ...rest] = x.shape.slice(0, -1);

    // Split into groups.
    x = x.reshape(batch, -1, this.numGroups, dims / this.numGroups);
    x = x.transpose(0, 1, 3, 2).reshape(batch, -1, this.numGroups);

    // Normalize.
    const means = mx.mean(x, 1, true);
    const variance = mx.variance(x, 1, true);
    x = mx.multiply(mx.subtract(x, means),
                    mx.rsqrt(mx.add(variance, this.eps)));
    x = x.reshape(batch, -1, dims / this.numGroups, this.numGroups);
    x = x.transpose(0, 1, 3, 2).reshape(batch, ...rest, dims);

    return x;
  }

  private groupNorm(x: mx.array) {
    const [dims] = x.shape.slice(-1);
    const [batch, ...rest] = x.shape.slice(0, -1);

    // Split into groups.
    x = x.reshape(batch, -1, this.numGroups);

    // Normalize.
    const means = mx.mean(x, 1, true)
    const variance = mx.variance(x, 1, true);
    x = mx.multiply(mx.subtract(x, means),
                    mx.rsqrt(mx.add(variance, this.eps)));
    x = x.reshape(batch, ...rest, dims);

    return x;
  }
}

/**
 * Applies Batch Normalization over a 2D or 3D input.
 *
 * @remarks
 *
 * The input shape is specified as `NC` or `NLC`, where `N` is the batch, `C`
 * is the number of features or channels, and `L` is the sequence length. The
 * output has the same shape as the input. For four-dimensional arrays, the
 * shape is `NHWC`, where `H` and `W` are the height and width respectively.
 *
 * ```math
 * y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,
 * ```
 *
 * where `gamma` and `beta` are learned per feature dimension
 * parameters initialized at 1 and 0 respectively.
 *
 * Reference: https://arxiv.org/abs/1502.03167
 *
 * @param numFeatures - The feature dimension to normalize over.
 * @param eps - A small additive constant for numerical stability.
 * Default: `1e-5`.
 * @param momentum - The momentum for updating the running mean and variance.
 * Default: `0.1`.
 * @param affine - If `true`, apply a learned affine transformation after the
 * normalization. Default: `true`.
 * @param trackRunningStats - If `true`, track the running mean and variance.
 * Default: `true`.
 *
 * @example
 * ```typescript
 * import {core as mx, nn} from '@frost-beta/mlx';
 *
 * const x = mx.random.normal([5, 4]);
 * const bn = new nn.BatchNorm(4);
 * const output = bn.forward(x);
 * ```
 */
export class BatchNorm extends Module {
  numFeatures: number;
  eps: number;
  momentum: number;
  trackRunningStats: boolean;
  weight?: mx.array;
  bias?: mx.array;
  runningMean?: mx.array;
  runningVar?: mx.array;

  constructor(numFeatures: number, eps = 1e-5, momentum = 0.1, affine = true, trackRunningStats = true) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.momentum = momentum;
    this.trackRunningStats = trackRunningStats;
    if (affine) {
      this.weight = mx.ones([numFeatures]);
      this.bias = mx.zeros([numFeatures]);
    }
    if (trackRunningStats) {
      this.runningMean = mx.zeros([numFeatures]);
      this.runningVar = mx.ones([numFeatures]);
      this.freeze(false, ['runningMean', 'runningVar']);
    }
  }

  override unfreeze(...args): this {
    super.unfreeze(...args);
    this.freeze(false, ['runningMean', 'runningVar']);
    return this;
  }

  override toStringExtra(): string {
    return `${this.numFeatures}, eps=${this.eps.toExponential()}, momentum=${this.momentum}, ` +
           `affine=${!!this.weight}, trackRunningStats=${this.trackRunningStats}`
  }

  // Forward pass of BatchNorm.
  forward(x: mx.array): mx.array {
    if (x.ndim < 2 || x.ndim > 4) {
      throw Error(`Expected input tensor to have 2, 3 or 4 dimensions, but got ${x.ndim}`);
    }

    // Calculate the mean and variance used to normalize the input x.
    let [mean, variance] = this.calcStats(x);
    if (this.training && this.trackRunningStats) {
      this.runningMean = mx.add(mx.multiply(1 - this.momentum, this.runningMean),
                                mx.multiply(this.momentum, mean));
      this.runningVar = mx.add(mx.multiply(1 - this.momentum, this.runningVar),
                               mx.multiply(this.momentum, variance));
    } else if (this.trackRunningStats) {
      mean = this.runningMean;
      variance = this.runningVar;
    }
    x = mx.multiply(mx.subtract(x, mean),
                    mx.rsqrt(mx.add(variance, this.eps)));
    if (this.weight)
      return mx.add(mx.multiply(this.weight, x), this.bias);
    else
      return x;
  }

  // Calculate the mean and variance of the input tensor across the batch and
  // spatial dimensions.
  private calcStats(x: mx.array): [mx.array, mx.array] {
    const reductionAxes = range(0, x.ndim - 1);
    return [mx.mean(x, reductionAxes), mx.variance(x, reductionAxes)];
  }
}
