import {core as mx} from '../../..';
import {Module} from './base';

/**
 * Applies the sigmoid function.
 *
 * @remarks
 *
 * ```math
 * \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
 * ```
 */
export const sigmoid = mx.compile((x: mx.array) => mx.sigmoid(x), true);

/**
 * Applies the Rectified Linear Unit.
 *
 * @remarks
 *
 * Simply `mx.maximum(x, 0)`.
 */
export const relu = mx.compile((x: mx.array) => mx.maximum(x, 0), true);

/**
 * Applies the Leaky Rectified Linear Unit.
 *
 * @remarks
 *
 * Simply `mx.maximum(negativeSlope * x, x)`.
 */
export const leakyRelu = mx.compile((x: mx.array, negativeSlope = 0.01) => {
  return mx.maximum(mx.multiply(negativeSlope, x), x)
}, true);

/**
 * Applies the Log Softmax function.
 *
 * @remarks
 *
 * Applies `x + log(sum(e^x[i]))` element wise.
 */
export const logSoftmax = mx.compile((x: mx.array, axis = -1) => {
  return mx.subtract(x, mx.logsumexp(x, axis, true));
}, true);

/**
 * Applies the Exponential Linear Unit.
 *
 * @remarks
 *
 * Simply `mx.where(x > 0, x, alpha * (mx.exp(x) - 1))`.
 */
export const elu = mx.compile((x: mx.array, alpha = 1.0) => {
  return mx.where(mx.greater(x, 0),
                  x,
                  mx.multiply(alpha,
                              mx.subtract(mx.exp(x), 1)));
}, true);

/**
 * Applies the Rectified Linear Unit 6.
 *
 * @remarks
 *
 * Applies `min(max(x, 0), 6.0)` element wise.
 */
export const relu6 = mx.compile((x: mx.array) => {
  return mx.minimum(mx.maximum(x, 0), 6.0);
}, true);

/**
 * Applies the Softmax function.
 *
 * @remarks
 *
 * Applies `mx.exp(x_i) / mx.sum(mx.exp(x_j))` element wise.
 */
export const softmax = mx.compile((x: mx.array, axis = -1) => {
  return mx.softmax(x, axis);
}, true);

/**
 * Applies the Softplus function.
 *
 * @remarks
 *
 * Applies `mx.log(1 + mx.exp(x))` element wise.
 */
export const softplus = mx.compile((x: mx.array) => {
  return mx.logaddexp(x, 0);
}, true);

/**
 * Applies the Softsign function.
 *
 * @remarks
 *
 * Applies `x / (1 + mx.abs(x))` element wise.
 */
export const softsign = mx.compile((x: mx.array) => {
  return mx.divide(x, mx.add(1, mx.abs(x)));
}, true);

/**
 * Applies the Softshrink activation function.
 *
 * @remarks
 *
 * ```math
 * \text{softshrink}(x) = \begin{cases}
 * x - \lambda & \text{if } x > \lambda \\
 * x + \lambda & \text{if } x < -\lambda \\
 * 0 & \text{otherwise}
 * \end{cases}
 * ```
 */
export const softshrink = mx.compile((x: mx.array, lambd: number = 0.5) => {
  return mx.where(mx.greater(mx.abs(x), lambd),
                  mx.subtract(x, mx.multiply(mx.sign(x), lambd)),
                  0);
}, true);

/**
 * Applies the Continuously Differentiable Exponential Linear Unit.
 *
 * @remarks
 *
 * Applies `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))` element wise.
 */
export const celu = mx.compile((x: mx.array, alpha = 1.0) => {
  return mx.add(
      mx.maximum(x, 0.0),
      mx.multiply(alpha,
                  mx.subtract(mx.exp(mx.divide(mx.minimum(x, 0.0),
                                               alpha)),
                              1)));
}, true);

/**
 * Applies the Sigmoid Linear Unit. Also known as Swish.
 *
 * @remarks
 *
 * Applies `x * mx.sigmoid(x)` element wise, where `mx.sigmoid(x)`
 * is the logistic sigmoid.
 */
export const silu = mx.compile((x: mx.array) => {
  return mx.multiply(x, mx.sigmoid(x));
}, true);

/**
 * Applies the Log Sigmoid function.
 *
 * @remarks
 *
 * Applies `log(sigma(x)) = -log(1 + e^{-x})` element wise.
 */
export const logSigmoid = mx.compile((x: mx.array) => {
  return mx.negative(softplus(mx.negative(x)));
}, true);

/**
 * Applies the Gaussian Error Linear Units function.
 *
 * @remarks
 *
 * The computation is done by:
 *
 * ```math
 * \textrm{GELU}(x) = x * \Phi(x)
 * ```
 *
 * where `phi(x)` is the Gaussian CDF.
 */
export const gelu = mx.compile((x: mx.array) => {
  return mx.divide(mx.multiply(x,
                               mx.add(1,
                                      mx.erf(mx.divide(x, Math.sqrt(2))))),
                   2);
}, true);

/**
 * An approximation to Gaussian Error Linear Unit.
 *
 * @remarks
 *
 * This function approximates `gelu` with a maximum absolute error `<
 * 0.0005` in the range `[-6, 6]` using the following formula:
 *
 * ```math
 * x = 0.5 * x * \left(1 + \text{Tanh}\left((\sqrt{2 / \pi} * \left(x + 0.044715 * x^3\right)\right)\right)
 * ```
 */
export const geluApprox = mx.compile((x: mx.array) => {
  return mx.multiply(
      0.5,
      mx.multiply(
          x,
          mx.add(
             1,
             mx.tanh(
                mx.multiply(
                   Math.sqrt(2 / Math.PI),
                   mx.add(x, mx.multiply(0.044715, mx.power(x, 3))))))));
}, true);

/**
 * A fast approximation to Gaussian Error Linear Unit.
 *
 * @remarks
 *
 * This function approximates `gelu` with a maximum absolute error :`<
 * 0.015` in the range :`[-6, 6]` using the following.
 *
 * ```math
 * x = x \sigma\left(1.702 x\right)
 * ```
 *
 * where `\sigma(\cdot)` is the logistic sigmoid.
 *
 * References:
 * - https://github.com/hendrycks/GELUs
 * - https://arxiv.org/abs/1606.08415
 */
export const geluFastApprox = mx.compile((x: mx.array) => {
  return mx.multiply(x, mx.sigmoid(mx.multiply(1.702, x)));
}, true);

/**
 * Applies the gated linear unit function.
 *
 * @remarks
 *
 * This function splits the `axis` dimension of the input into two halves
 * (`a` and `b`) and applies `a * \sigma(b)`.
 *
 * @param axis - The dimension to split along. Default: `-1`
 */
export function glu(x: mx.array, axis: number = -1): mx.array {
  const [a, b] = mx.split(x, 2, axis);
  return mx.multiply(a, mx.sigmoid(b));
}

/**
 * Applies the Step Activation Function.
 *
 * @remarks
 *
 * This function implements a binary step activation, where the output is set
 * to 1 if the input is greater than a specified threshold, and 0 otherwise.
 *
 * ```math
 * \text{step}(x) = \begin{cases}
 * 0 & \text{if } x < \text{threshold} \\
 * 1 & \text{if } x \geq \text{threshold}
 * \end{cases}
 *
 * @param threshold - The value to threshold at.
 */
export const step = mx.compile((x: mx.array, threshold = 0.0) => {
  return mx.where(mx.greater(x, threshold), 1, 0);
}, true);

/**
 * Applies the Scaled Exponential Linear Unit.
 *
 * @remarks
 *
 * ```math
 * \text{selu}(x) = \begin{cases}
 * \lambda x & \text{if } x > 0 \\
 * \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
 * \end{cases}
 * ```
 *
 * where `lambda = 1.0507` and `alpha = 1.67326`.
 */
export const selu = mx.compile((x: mx.array) => {
  return mx.multiply(elu(x, 1.67326), 1.0507);
}, true);

/**
 * Applies the element-wise parametric ReLU.
 *
 * @remarks
 *
 * ```math
 * \text{PReLU}(x) = \max(0,x) + a * \min(0,x)
 * ```
 *
 * where `a` is an array.
 */
export const prelu = mx.compile((x: mx.array, alpha: mx.array) => {
  return mx.add(mx.maximum(0, x), mx.multiply(alpha, mx.minimum(0, x)));
}, true);

/**
 * Applies the Mish function, element-wise.
 *
 * @remarks
 *
 * Mish: A Self Regularized Non-Monotonic Neural Activation Function.
 *
 * Reference: https://arxiv.org/abs/1908.08681
 *
 * ```math
 * \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
 * ```
 */
export const mish = mx.compile((x: mx.array) => {
  return mx.multiply(x, mx.tanh(softplus(x)));
}, true);

/**
 * Applies the hardswish function, element-wise.
 *
 * @remarks
 *
 * ```math
 * \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
 * ```
 */
export const hardswish = mx.compile((x: mx.array) => {
  const max_x_3 = mx.maximum(mx.add(x, 3), 0);
  return mx.divide(mx.multiply(x, mx.minimum(max_x_3, 6)), 6);
}, true);

/**
 * Applies the hyperbolic tangent function.
 *
 * @remarks
 *
 * Simply `mx.tanh(x)`.
 */
export function tanh(x: mx.array) {
  return mx.tanh(x);
}

/**
 * Applies the gated linear unit function.
 *
 * @remarks
 *
 * This function splits the `axis` dimension of the input into two halves
 * (`a` and `b`) and applies `a * \sigma(b)`.
 *
 * ```math
 * \textrm{GLU}(x) = a * \sigma(b)
 * ```
 *
 * @param axis - The dimension to split along. Default: `-1`
 */
export class GLU extends Module {
  axis: number;

  constructor(axis: number = -1) {
    super();
    this.axis = axis;
  }

  override forward(x: mx.array): mx.array {
    return glu(x, this.axis);
  }
}

/**
 * Applies the sigmoid function, element-wise.
 *
 * @remarks
 *
 * ```math
 * \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
 * ```
 */
export class Sigmoid extends Module {
  override forward(x: mx.array): mx.array {
    return sigmoid(x);
  }
}

/**
 * Applies the Mish function, element-wise.
 *
 * @remarks
 *
 * Reference: https://arxiv.org/abs/1908.08681
 *
 * ```math
 * \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
 * ```
 */
export class Mish extends Module {
  override forward(x: mx.array): mx.array {
    return mish(x);
  }
}

/**
 * Applies the Rectified Linear Unit.
 *
 * @remarks
 *
 * Simply `mx.maximum(x, 0)`.
 */
export class ReLU extends Module {
  override forward(x: mx.array): mx.array {
    return relu(x);
  }
}

/**
 * Applies the Leaky Rectified Linear Unit.
 *
 * Simply `mx.maximum(negativeSlope * x, x)`.
 *
 * @param negativeSlope - Controls the angle of the negative slope. Default: `1e-2`
 */
export class LeakyReLU extends Module {
  negativeSlope: number;

  constructor(negativeSlope = 1e-2) {
    super();
    this.negativeSlope = negativeSlope;
  }

  override forward(x: mx.array): mx.array {
    return leakyRelu(x, this.negativeSlope);
  }
}

/**
 * Applies the Exponential Linear Unit.
 *
 * Simply `mx.where(x > 0, x, alpha * (mx.exp(x) - 1))`.
 *
 * @param alpha - The `alpha` value for the ELU formulation. Default: `1.0`
 */
export class ELU extends Module {
  alpha: number;

  constructor(alpha = 1.0) {
    super();
    this.alpha = alpha;
  }

  override forward(x: mx.array): mx.array {
    return elu(x, this.alpha);
  }
}

/**
 * Applies the Rectified Linear Unit 6.
 */
export class ReLU6 extends Module {
  override forward(x: mx.array): mx.array {
    return relu6(x);
  }
}

/**
 * Applies the Softmax function.
 */
export class Softmax extends Module {
  override forward(x: mx.array): mx.array {
    return softmax(x);
  }
}

/**
 * Applies the Softplus function.
 */
export class Softplus extends Module {
  override forward(x: mx.array): mx.array {
    return softplus(x);
  }
}

/**
 * Applies the Softsign function.
 */
export class Softsign extends Module {
  override forward(x: mx.array): mx.array {
    return softsign(x);
  }
}

/**
 * Applies the Softshrink function.
 *
 * @param lambd - The `lambda` value for Softshrink. Default: `0.5`.
 */
export class SoftShrink extends Module {
  lambd: number;

  constructor(lambd = 0.5) {
    super();
    this.lambd = lambd;
  }

  override forward(x: mx.array): mx.array {
    return softshrink(x, this.lambd);
  }
}

/**
 * Applies the Continuously Differentiable Exponential Linear Unit.
 *
 * @remarks
 *
 * Applies `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))` element wise.
 *
 * @param alpha - The `alpha` value for the CELU formulation. Default: `1.0`.
 */
export class CELU extends Module {
  alpha: number;

  constructor(alpha = 1.0) {
    super();
    this.alpha = alpha;
  }

  override forward(x: mx.array): mx.array {
    return celu(x, this.alpha);
  }
}

/**
 * Applies the Sigmoid Linear Unit. Also known as Swish.
 */
export class SiLU extends Module {
  override forward(x: mx.array): mx.array {
    return silu(x);
  }
}

/**
 * Applies the Log Softmax function.
 */
export class LogSoftmax extends Module {
  override forward(x: mx.array): mx.array {
    return logSoftmax(x);
  }
}

/**
 * Applies the Log Sigmoid function.
 */
export class LogSigmoid extends Module {
  override forward(x: mx.array): mx.array {
    return logSigmoid(x);
  }
}

/**
 * Applies the element-wise parametric ReLU.
 *
 * @remarks
 *
 * Applies `max(0, x) + a * min(0, x)` element wise, where `a`
 * is an array.
 */
export class PReLU extends Module {
  weight: mx.array;

  constructor(numParameters = 1, init = 0.25) {
    super();
    this.weight = mx.full([numParameters], init);
  }

  override forward(x: mx.array): mx.array {
    return prelu(x, this.weight);
  }
}

/**
 * Applies the Gaussian Error Linear Units.
 *
 * @remarks
 *
 * ```math
 * GELU(x) = x * \Phi(x)
 * ```
 *
 * where `\Phi(x)` is the Gaussian CDF.
 *
 * However, if `approx` is set to 'precise' or 'fast' it applies
 *
 * ```math
 * GELUApprox(x) = 0.5 * x * (1 + Tanh((sqrt(2 / \pi) * (x + 0.044715 * x^3)))
 * GELUFast(x) = x * \sigma(1.773 * x)
 * ```
 *
 * respectively.
 *
 * @param approx - Which approximation to gelu to use if any. Choices: `'none' | 'precise' | 'fast'`
 */
export class GELU extends Module {
  private act: (x: mx.array) => mx.array;

  constructor(approx: 'none' | 'precise' | 'fast' = 'none') {
    super();
    switch (approx) {
      case 'none':
        this.act = gelu;
        break;
      case 'precise':
        this.act = geluApprox;
        break;
      case 'fast':
        this.act = geluFastApprox;
        break;
      default:
        throw new Error(`The approximation should be in ['none', 'precise', 'fast'] but '${approx}' was given`);
    }
  }

  override forward(x: mx.array): mx.array {
    return this.act(x);
  }
}

/**
 * Applies the hyperbolic tangent function.
 */
export class Tanh extends Module {
  override forward(x: mx.array): mx.array {
    return tanh(x);
  }
}

/**
 * Applies the hardswish function, element-wise.
 */
export class Hardswish extends Module {
  override forward(x: mx.array): mx.array {
    return hardswish(x);
  }
}

/**
 * Applies the Step Activation Function.
 *
 * @remarks
 *
 * This function implements a binary step activation, where the output is set
 * to 1 if the input is greater than a specified threshold, and 0 otherwise.
 *
 * ```math
 * \text{step}(x) =
 * \begin{cases}
 * 0 & \text{if } x < \text{threshold} \\
 * 1 & \text{if } x \geq \text{threshold}
 * \end{cases}
 * ```
 *
 * @param threshold - The value to threshold at.
 */
export class Step extends Module {
  threshold: number;

  constructor(threshold: number = 0.0) {
    super();
    this.threshold = threshold;
  }

  override forward(x: mx.array): mx.array {
    return step(x, this.threshold);
  }
}

/**
 * Applies the Scaled Exponential Linear Unit.
 */
export class SELU extends Module {
  override forward(x: mx.array): mx.array {
    return selu(x);
  }
}
