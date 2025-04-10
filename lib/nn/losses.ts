import {core as mx} from '../..';
import {deepEqual} from '../utils';

type Reduction = 'none' | 'mean' | 'sum';

/**
 * Computes the cross entropy loss.
 *
 * @param logits - The unnormalized logits.
 * @param targets - The ground truth values. These can be class indices or
 * probabilities for each class. If the `targets` are class indices, then
 * `targets` shape should match the `logits` shape with the `axis` dimension
 * removed. If the `targets` are probabilities (or one-hot encoded), then the
 * `targets` shape should be the same as the `logits` shape.
 * @param weights - Optional weights for each target. Default: `undefined`.
 * @param axis - The axis over which to compute softmax. Default: `-1`.
 * @param labelSmoothing - Label smoothing factor. Default: `0`.
 * @param reduction - Specifies the reduction to apply to the output: `'none'` | 
 * `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed cross entropy loss.
 *
 * @example
 * ```typescript
 * import {core as mx, nn} from '@frost-beta/mlx';
 *
 * // Class indices as targets
 * const logits = mx.array([[2.0, -1.0], [-1.0, 2.0]]);
 * const targets = mx.array([0, 1]);
 * nn.losses.crossEntropy(logits, targets);
 * // array([0.0485873, 0.0485873], dtype=float32)
 *
 * // Probabilities (or one-hot vectors) as targets
 * logits = mx.array([[2.0, -1.0], [-1.0, 2.0]]);
 * targets = mx.array([[0.9, 0.1], [0.1, 0.9]]);
 * nn.losses.crossEntropy(logits, targets);
 * // array([0.348587, 0.348587], dtype=float32)
 * ```
 */
export function crossEntropy(logits: mx.array,
                             targets: mx.array,
                             weights?: mx.array,
                             axis = -1,
                             labelSmoothing = 0.0,
                             reduction: Reduction = 'none'): mx.array {
  if (labelSmoothing < 0 || labelSmoothing >= 1) {
    throw Error(`Label smoothing must in [0, 1), got ${labelSmoothing}.`);
  }

  // Whether targets are class indices or probabilities.
  const targetsAsProbs = targets.ndim === logits.ndim;

  function dropDim(shape: number[], axis: number): number[] {
    shape = shape.slice();
    shape.splice(axis);
    return shape;
  }

  // Check shapes in two cases:
  // targets as class indices and targets as probabilities.
  if ((targetsAsProbs && !deepEqual(targets.shape, logits.shape)) ||
      (!targetsAsProbs && !deepEqual(targets.shape, dropDim(logits.shape, axis)))) {
    throw Error(`Targets shape ${targets.shape} does not match logits shape ${logits.shape}.`);
  }

  let score;
  if (targetsAsProbs) {
    score = mx.sum(mx.multiply(logits, targets), axis);
  } else {
    score = mx.takeAlongAxis(logits, targets.index('...', null), axis).squeeze(-1);
  }

  const logsumexpLogits = mx.logsumexp(logits, axis);
  let loss;
  if (labelSmoothing > 0) {
    // Adjust the true class score with label smoothing.
    const adjustedScore = mx.multiply((1 - labelSmoothing), score);

    // Calculate the mean logit across the classes for smoothed loss.
    const meanLogits = mx.mean(logits, axis);
    const smoothedLoss = mx.multiply(mx.negative(meanLogits), labelSmoothing);

    // Combine the adjusted score and smoothed loss with the logsumexp logits.
    loss = mx.add(mx.subtract(logsumexpLogits, adjustedScore), smoothedLoss);
  } else {
    loss = mx.subtract(logsumexpLogits, score);
  }

  // Apply weights if provided.
  if (weights) {
    if (!deepEqual(weights.shape, loss.shape)) {
      throw Error(`Weights with shape ${weights.shape} is not the same as output loss with shape ${loss.shape}.`);
    }
    loss = mx.multiply(loss, weights);
  }

  // Apply reduction.
  return reduce(loss, reduction);
}

/**
 * Computes the binary cross entropy loss.
 *
 * @remarks
 *
 * By default, this function takes the pre-sigmoid logits, which results in a
 * faster and more precise loss. For improved numerical stability when
 * `withLogits: false`, the loss calculation clips the input probabilities
 * (in log-space) to a minimum value of ``-100``.
 *
 * @param inputs - The predicted values. If `withLogits` is `true`, then
 * `inputs` are unnormalized logits. Otherwise, `inputs` are probabilities.
 * @param targets - The binary target values in {0, 1}.
 * @param withLogits - Whether `inputs` are logits. Default: `true`.
 * @param weights - Optional weights for each target. Default: `undefined`.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'mean'`.
 *
 * @returns The computed binary cross entropy loss.
 *
 * @example
 * ```typescript
 * import {core as mx, nn} from '@frost-beta/mlx';
 *
 * const logits = mx.array([0.105361, 0.223144, 1.20397, 0.916291]);
 * const targets = mx.array([0, 0, 1, 1]);
 * const loss = nn.losses.binaryCrossEntropy(logits, targets, 'mean');
 * // loss: array(0.539245, dtype=float32)
 *
 * const probs = mx.array([0.1, 0.1, 0.4, 0.4]);
 * const targets = mx.array([0, 0, 1, 1]);
 * const loss = nn.losses.binaryCrossEntropy(probs, targets, false, undefined, 'mean');
 * // loss: array(0.510826, dtype=float32)
 * ```
 */
export function binaryCrossEntropy(inputs: mx.array,
                                   targets: mx.array,
                                   weights?: mx.array,
                                   withLogits = true,
                                   reduction: Reduction = 'mean'): mx.array {
  if (!deepEqual(inputs.shape, targets.shape)) {
    throw Error(`Inputs shape ${inputs.shape} does not match targets shape ${targets.shape}.`);
  }

  let loss;
  if (withLogits) {
    loss = mx.subtract(mx.logaddexp(0.0, inputs),
                       mx.multiply(inputs, targets));
  } else {
    const logInputsClip = mx.clip(mx.log(inputs), -100, undefined);
    const logInputsInvClip = mx.clip(mx.log(mx.subtract(1, inputs)), -100, undefined);
    loss = mx.negative(mx.add(mx.multiply(targets, logInputsClip),
                              mx.multiply(mx.subtract(1, targets), logInputsInvClip)));
  }

  // Apply weights if provided.
  if (weights) {
    if (!deepEqual(weights.shape, loss.shape)) {
      throw Error(`Weights with shape ${weights.shape} is not the same as output loss with shape ${loss.shape}.`);
    }
    loss = mx.multiply(loss, weights);
  }

  return reduce(loss, reduction);
}

/**
 * Computes the L1 loss.
 *
 * @param predictions - The predicted values.
 * @param targets - The target values.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'mean'`.
 *
 * @returns The computed L1 loss.
 */
export function l1Loss(predictions: mx.array,
                       targets: mx.array,
                       reduction: Reduction = 'mean'): mx.array {
  if (!deepEqual(predictions.shape, targets.shape)) {
    throw Error(`Predictions shape ${predictions.shape} does not match targets shape ${targets.shape}.`);
  }

  const loss = mx.abs(mx.subtract(predictions, targets));

  return reduce(loss, reduction);
}


/**
 * Computes the mean squared error loss.
 *
 * @param predictions - The predicted values.
 * @param targets - The target values.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'mean'`.
 *
 * @returns The computed mean squared error loss.
 */
export function mseLoss(predictions: mx.array,
                        targets: mx.array,
                        reduction: Reduction = 'mean'): mx.array {
  if (!deepEqual(predictions.shape, targets.shape)) {
    throw Error(`Predictions shape ${predictions.shape} does not match targets shape ${targets.shape}.`);
  }

  const loss = mx.square(mx.subtract(predictions, targets));

  return reduce(loss, reduction);
}

/**
 * Computes the negative log likelihood loss.
 *
 * @param inputs - The predicted distribution in log space.
 * @param targets - The target values.
 * @param axis - The distribution axis. Default: `-1`.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed NLL loss.
 *
 */
export function nllLoss(inputs: mx.array,
                        targets: mx.array,
                        axis = -1,
                        reduction: Reduction = 'none'): mx.array {
  const loss = mx.negative(mx.takeAlongAxis(inputs, targets.index('...', null), axis).squeeze(-1));

  return reduce(loss, reduction);
}

/**
 * Computes the negative log likelihood loss for a Gaussian distribution.
 *
 * @remarks
 *
 * The loss is given by:
 *
 * ```math
 * \frac{1}{2}\left(\log\left(\max\left(\text{vars},
 * \ \epsilon\right)\right) + \frac{\left(\text{inputs} - \text{targets} \right)^2}
 * {\max\left(\text{vars}, \ \epsilon \right)}\right) + \text{const.}
 * ```
 *
 * where `inputs` are the predicted means and `vars` are the predicted variances.
 *
 * @param inputs - The predicted expectation of the Gaussian distribution.
 * @param targets - The target values (samples from the Gaussian distribution).
 * @param vars - The predicted variance of the Gaussian distribution.
 * @param full - Whether to include the constant term in the loss calculation.
 *   Default: `false`.
 * @param eps - Small positive constant for numerical stability. Default: `1e-6`.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The Gaussian NLL loss.
 */
export function gaussianNllLoss(inputs: mx.array,
                                targets: mx.array,
                                vars: mx.array,
                                full = false,
                                eps = 1e-6,
                                reduction: Reduction = 'mean'): mx.array {
  if (!deepEqual(inputs.shape, targets.shape)) {
    throw Error(`Inputs shape ${inputs.shape} does not match targets shape ${targets.shape}.`);
  }

  if (!deepEqual(inputs.shape, vars.shape)) {
    throw Error(`Inputs shape ${inputs.shape} does not match vars shape ${vars.shape}.`);
  }

  // For stability.
  vars = mx.maximum(vars, eps);
  let loss = mx.multiply(0.5, mx.add(mx.log(vars), mx.divide(mx.square(mx.subtract(targets, inputs)), vars)));

  if (full) {
    loss = mx.add(loss, 0.5 * Math.log(2 * Math.PI));
  }

  return reduce(loss, reduction);
}

/**
 * Computes the Kullback-Leibler divergence loss.
 *
 * @remarks
 *
 * Computes the following when `reduction == 'none'`:
 *
 * ```typescript
 * mx.multiply(mx.exp(targets), mx.subtract(targets, inputs)).sum(axis)
 * ```
 *
 * @param inputs - Log probabilities for the predicted distribution.
 * @param targets - Log probabilities for the target distribution.
 * @param axis - The distribution axis. Default: `-1`.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed Kullback-Leibler divergence loss.
 */
export function klDivLoss(inputs: mx.array,
                          targets: mx.array,
                          axis = -1,
                          reduction: Reduction = 'none'): mx.array {
  const loss = mx.sum(mx.multiply(mx.exp(targets),
                                  mx.subtract(targets, inputs)),
                      axis);

  return reduce(loss, reduction);
}

/**
 * Computes the smooth L1 loss.
 *
 * @remarks
 *
 * The smooth L1 loss is a variant of the L1 loss which replaces the absolute
 * difference with a squared difference when the absolute difference is less
 * than `beta`.
 *
 * The formula for the smooth L1 Loss is:
 *
 * ```math
 * l = \begin{cases}
 *       0.5 (x - y)^2, & \text{if } (x - y) < \beta \\
 *       |x - y| - 0.5 \beta, & \text{otherwise}
 *     \end{cases}
 * ```
 *
 * @param predictions - Predicted values.
 * @param targets - Ground truth values.
 * @param beta - The threshold after which the loss changes
 *   from the squared to the absolute difference. Default: `1.0`.
 * @param reduction - Specifies the reduction to apply to the output:
 *   `'none'` | `'mean'` | `'sum'`. Default: `'mean'`.
 *
 * @returns The computed smooth L1 loss.
 */
export function smoothL1Loss(predictions: mx.array,
                             targets: mx.array,
                             beta = 1.0,
                             reduction: Reduction = 'mean'): mx.array {
  if (!deepEqual(predictions.shape, targets.shape)) {
    throw Error(`Predictions shape ${predictions.shape} does not match targets shape ${targets.shape}.`);
  }

  let diff = mx.abs(mx.subtract(predictions, targets));
  let loss = mx.where(mx.less(diff, beta),
                      mx.multiply(0.5, mx.divide(mx.square(diff), beta)),
                      mx.subtract(mx.abs(diff), 0.5 * beta));

  return reduce(loss, reduction);
}

/**
 * Computes the triplet loss for a set of anchor, positive, and negative samples.
 * Margin is represented with alpha in the math section.
 *
 * @remark
 *
 * ```math
 * max(||A - P||_p - ||A - N||_p + alpha, 0)
 * ```
 *
 * @param anchors - The anchor samples.
 * @param positives - The positive samples.
 * @param negatives - The negative samples.
 * @param axis - The distribution axis. Default: `-1`.
 * @param p - The norm degree for pairwise distance. Default: `2`.
 * @param margin - Margin for the triplet loss. Defaults to `1.0`.
 * @param eps - Small positive constant to prevent numerical instability. Defaults to `1e-6`. 
 * @param reduction - Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns Computed triplet loss. If reduction is "none", returns a tensor of the same shape as input;
 * if reduction is "mean" or "sum", returns a scalar tensor.
 */
export function tripletLoss(anchors: mx.array,
                            positives: mx.array,
                            negatives: mx.array,
                            axis = -1,
                            p = 2,
                            margin = 1.0,
                            eps = 1e-6,
                            reduction: Reduction = 'none'): mx.array {
  const loss = mx.maximum(
    mx.add(mx.subtract(mx.sqrt(mx.add(mx.power(mx.subtract(anchors, positives), p).sum(axis),
                                      eps)),
                       mx.sqrt(mx.add(mx.power(mx.subtract(anchors, negatives), p).sum(axis),
                                      eps))),
           margin),
    0);

  return reduce(loss, reduction);
}

/**
 * Computes the hinge loss between inputs and targets.
 *
 * @remarks
 *
 * The hinge loss is computed as:
 *
 * ```math
 * \text{hinge}(y, y_{\text{pred}}) = \max(0, 1 - y \cdot y_{\text{pred}})
 * ```
 *
 * @param inputs - The predicted values.
 * @param targets - The target values. They should be -1 or 1.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed hinge loss.
 */
export function hingeLoss(inputs: mx.array, targets: mx.array, reduction: Reduction = 'none'): mx.array {
  const loss = mx.maximum(mx.subtract(1, mx.multiply(inputs, targets)), 0);

  return reduce(loss, reduction);
}

/**
 * Computes the Huber loss between inputs and targets.
 *
 * @remarks
 *
 * ```math
 * l_{\delta}(a) =
 * \left\{ \begin{array}{ll}
 *   \frac{1}{2} a^2 & \text{for } |a| \leq \delta, \\
 *   \delta \left( |a| - \frac{1}{2} \delta \right) & \text{otherwise.}
 * \end{array} \right.
 * ```
 *
 * @param inputs - The predicted values.
 * @param targets - The target values.
 * @param delta - The threshold at which to change between L1 and L2 loss.
 *   Default: `1.0`.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed Huber loss.
 */
export function huberLoss(inputs: mx.array,
                          targets: mx.array,
                          delta = 1.0,
                          reduction: Reduction = 'none'): mx.array {
  const errors = mx.subtract(inputs, targets);
  const absErrors = mx.abs(errors);
  const quadratic = mx.minimum(absErrors, delta);
  const linear = mx.subtract(absErrors, quadratic);
  const loss = mx.add(mx.multiply(0.5, mx.power(quadratic, 2)),
                      mx.multiply(delta, linear));

  return reduce(loss, reduction);
};

/**
 * Computes the log cosh loss between inputs and targets.
 *
 * @remarks
 *
 * Logcosh acts like L2 loss for small errors, ensuring stable gradients,
 * and like the L1 loss for large errors, reducing sensitivity to outliers. This
 * dual behavior offers a balanced, robust approach for regression tasks.
 *
 * ```math
 * \text{logcosh}(y_{\text{true}}, y_{\text{pred}}) =
 *   \frac{1}{n} \sum_{i=1}^{n}
 *   \log(\cosh(y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)}))
 * ```
 *
 * @param inputs - The predicted values.
 * @param targets - The target values.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed log cosh loss.
 */
export function logCoshLoss(inputs: mx.array,
                            targets: mx.array,
                            reduction: Reduction = 'none'): mx.array {
  const errors = mx.subtract(inputs, targets);
  const loss = mx.subtract(mx.logaddexp(errors, mx.multiply(-1, errors)),
                           Math.log(2));

  return reduce(loss, reduction);
};

/**
 * Computes the cosine similarity between the two inputs.
 *
 * @remarks
 *
 * The cosine similarity loss is given by
 *
 * ```math
 * \frac{x_1 \cdot x_2}{\max(\|x_1\|  \cdot \|x_2\|, \epsilon)}
 * ```
 *
 * @param x1 - The first set of inputs.
 * @param x2 - The second set of inputs.
 * @param axis - The embedding axis. Default: `1`.
 * @param eps - The minimum value of the denominator used for numerical stability. Default: `1e-8`.
 * @param reduction - Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed cosine similarity loss.
 */
export function cosineSimilarityLoss(x1: mx.array,
                                     x2: mx.array,
                                     axis = 1,
                                     eps = 1e-8,
                                     reduction: Reduction = 'none'): mx.array {
  const x1Norm = mx.linalg.norm(x1, undefined, axis);
  const x2Norm = mx.linalg.norm(x2, undefined, axis);

  let loss = mx.divide(mx.sum(mx.multiply(x1, x2), axis),
                       mx.maximum(mx.multiply(x1Norm, x2Norm), eps));

  return reduce(loss, reduction);
}

/**
 * Calculate the margin ranking loss that loss given inputs `x1`, `x2` and a
 * label `y` (containing 1 or -1).
 *
 * @remarks
 *
 * The loss is given by:
 *
 * ```math
 * \text{loss} = \max (0, -y * (x_1 - x_2) + \text{margin})
 * ```
 *
 * Where `y` represents `targets`, `x_1` represents `inputs1` and `x_2`
 * represents `inputs2`.
 *
 * @param inputs1 - Scores for the first input.
 * @param inputs2 - Scores for the second input.
 * @param targets - Labels indicating whether samples in `inputs1` should be
 * ranked higher than samples in `inputs2`. Values should be 1 or -1.
 * @param margin - The margin by which the scores should be separated.
 *   Default: `0.0`.
 * @param reduction - Specifies the reduction to apply to the output:
 * `'none'` | `'mean'` | `'sum'`. Default: `'none'`.
 *
 * @returns The computed margin ranking loss.
 */
export function marginRankingLoss(inputs1: mx.array,
                                  inputs2: mx.array,
                                  targets: mx.array,
                                  margin = 0.0,
                                  reduction: Reduction = 'none'): mx.array {
  if (!deepEqual(inputs1.shape, inputs2.shape) ||
      !deepEqual(inputs1.shape, targets.shape)) {
    throw Error(`The shapes of the arguments do not match. The provided shapes are inputs1.shape=${inputs1.shape}, inputs2.shape=${inputs2.shape}, and targets.shape=${targets.shape}.`);
  }

  const differences = mx.subtract(inputs1, inputs2);
  const loss = mx.maximum(mx.add(mx.multiply(mx.negative(targets), differences),
                                 margin),
                          0);

  return reduce(loss, reduction);
}

// Helpers.
const reduce = (loss: mx.array, reduction: Reduction = 'none'): mx.array => {
  if (reduction === 'mean') {
    return mx.mean(loss);
  } else if (reduction === 'sum') {
    return mx.sum(loss);
  } else if (reduction === 'none') {
    return loss;
  } else {
    throw new Error("Invalid reduction. Must be 'none', 'mean', or 'sum'.");
  }
};
