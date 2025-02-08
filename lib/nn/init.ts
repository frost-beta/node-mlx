import {core as mx} from '../..';

/**
 * An initializer that returns an array filled with `value`.
 *
 * @param value - The value to fill the array with.
 * @param dtype - The data type of the array. Default is `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with `value`.
 *
 * @example
 * ```typescript
 * const initFn = constant(0.5);
 * initFn(mx.zeros([2, 2]));
 * // array([[0.5, 0.5],
 * //       [0.5, 0.5]], dtype=float32)
 * ```
 */
export function constant(value: number, dtype: mx.Dtype = mx.float32): (a: mx.array) => mx.array {
  const initializer = (a: mx.array): mx.array => {
    return mx.full(a.shape, value, dtype);
  }
  return initializer;
}

/**
 * An initializer that returns samples from a normal distribution.
 *
 * @param mean - Mean of the normal distribution. Default is `0.0`.
 * @param std - Standard deviation of the normal distribution. Default is `1.0`.
 * @param dtype - The data type of the array. Default is `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with samples from a normal distribution.
 *
 * @example
 * ```typescript
 * const initFn = normal();
 * initFn(mx.zeros([2, 2]));
 * // array([[-0.982273, -0.534422],
 * //       [0.380709, 0.0645099]], dtype=float32)
 * ```
 */
export function normal(mean: number = 0.0, std: number = 1.0, dtype: mx.Dtype = mx.float32): (a: mx.array) => mx.array {
  const initializer = (a: mx.array): mx.array => {
    return mx.random.normal(a.shape, dtype, mean, std);
  }
  return initializer;
}

/**
 * An initializer that returns samples from a uniform distribution.
 *
 * @param low - The lower bound of the uniform distribution. Default is `0.0`.
 * @param high - The upper bound of the uniform distribution. Default is `1.0`.
 * @param dtype - The data type of the array. Default is `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with samples from a uniform distribution.
 *
 * @example
 * ```typescript
 * const initFn = uniform(0, 1);
 * initFn(mx.zeros([2, 2]));
 * // array([[0.883935, 0.863726],
 * //       [0.617261, 0.417497]], dtype=float32)
 * ```
 */
export function uniform(low: number = 0.0, high: number = 1.0, dtype: mx.Dtype = mx.float32): (a: mx.array) => mx.array {
  const initializer = (a: mx.array): mx.array => {
    return mx.random.uniform(low, high, a.shape, dtype);
  }
  return initializer;
}

/**
 * An initializer that returns an identity matrix.
 *
 * @param dtype - The data type of the array. Default is `mx.float32`.
 *
 * @returns An initializer that returns an identity matrix with the same shape as the input.
 *
 * @example
 * ```typescript
 * const initFn = identity();
 * initFn(mx.zeros([2, 2]));
 * // array([[1, 0],
 * //       [0, 1]], dtype=float32)
 * ```
 */
export function identity(dtype: mx.Dtype = mx.float32): (arr: mx.array) => mx.array {
  const initializer = (arr: mx.array): mx.array => {
    if (arr.ndim != 2 || arr.shape[0] != arr.shape[1]) {
      throw new Error(`The input array must be a square matrix but got shape ${arr.shape}.`);
    }
    return mx.eye(arr.shape[0], null, null, dtype);
  }
  return initializer;
}

/**
 * A Glorot normal initializer.
 *
 * @remarks
 *
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (`fan_in`) and output
 * (`fan_out`) units according to:
 *
 * ```math
 * \sigma = \gamma \sqrt{\frac{2.0}{\text{fanIn} + \text{fanOut}}}
 * ```
 *
 * For more details see the original reference: [Understanding the difficulty
 * of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
 *
 * @param dtype - The data type of the array. Default is `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with samples from the Glorot normal distribution.
 *
 * @example
 * ```typescript
 * const initFn = glorotNormal();
 * initFn(mx.zeros([2, 2]));
 * // array([[0.191107, 1.61278],
 * //       [-0.150594, -0.363207]], dtype=float32)
 * initFn(mx.zeros([2, 2]), 4.0);
 * // array([[1.89613, -4.53947],
 * //       [4.48095, 0.995016]], dtype=float32)
 * ```
 */
export function glorotNormal(dtype: mx.Dtype = mx.float32): (a: mx.array, gain?: number) => mx.array {
  const initializer = (a: mx.array, gain = 1.0): mx.array => {
    const [fanIn, fanOut] = calculateFanInFanOut(a);
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    return mx.random.normal(a.shape, dtype, null, std);
  }
  return initializer;
}

/**
 * A Glorot uniform initializer.
 *
 * @remarks
 *
 * This initializer samples from a uniform distribution with a range computed
 * from the number of input (`fanIn`) and output (`fanOut`) units according to:
 *
 * ```math
 * \sigma = \gamma \sqrt{\frac{6.0}{\text{fanIn} + \text{fanOut}}}
 * ```
 *
 * For more details see the original reference: [Understanding the difficulty
 * of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
 *
 * @param dtype - The data type of the array. Default: `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with samples from the Glorot uniform distribution.
 *
 * @example
 * ```typescript
 * const initFn = glorotUniform();
 * initFn(mx.zeros([2, 2]));
 * // array([[0.223404, -0.890597],
 * //       [-0.379159, -0.776856]], dtype=float32)
 * initFn(mx.zeros([2, 2]), 4.0);
 * // array([[-1.90041, 3.02264],
 * //       [-0.912766, 4.12451]], dtype=float32)
 * ```
 */
export function glorotUniform(dtype: mx.Dtype = mx.float32): (a: mx.array, gain?: number) => mx.array {
  const initializer = (a: mx.array, gain: number = 1.0): mx.array => {
    const [fanIn, fanOut] = calculateFanInFanOut(a);
    const limit = gain * Math.sqrt(6.0 / (fanIn + fanOut));
    return mx.random.uniform(-limit, limit, a.shape, dtype);
  }
  return initializer;
}

/**
 * Build a He normal initializer.
 *
 * @remarks
 *
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (`fanIn`) or output
 * (`fanOut`) units according to:
 *
 * ```math
 * \sigma = \gamma \frac{1}{\sqrt{\text{fan}}}
 * ```
 *
 * where `fan` is either the number of input units when the `mode` is `"fanIn"`
 * or output units when the `mode` is `"fanOut"`.
 *
 * For more details see the original reference: [Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
 *
 * @param dtype - The data type of the array. Default: `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with samples from the He normal distribution.
 *
 * @example
 * ```
 * const initFn = nn.init.heNormal();
 * initFn(mx.zeros([2, 2]));  // uses fan_in
 * // [[-1.25211, 0.458835],
 * //  [-0.177208, -0.0137595]], dtype=float32)
 * initFn(mx.zeros([2, 2]), 'fanOut', 5);
 * // [[5.6967, 4.02765],
 * //  [-4.15268, -2.75787]], dtype=float32)
 * ```
 */
export function heNormal(dtype: mx.Dtype = mx.float32): (a: mx.array, mode?: 'fanIn' | 'fanOut', gain?: number) => mx.array {
  const initializer = (a: mx.array, mode: 'fanIn' | 'fanOut' = 'fanIn', gain: number = 1.0): mx.array => {
    const [fanIn, fanOut] = calculateFanInFanOut(a);
    let fan;
    if (mode === 'fanIn') {
      fan = fanIn;
    } else if (mode === 'fanOut') {
      fan = fanOut;
    } else {
      throw new Error(`Invalid mode: ${mode}. Valid modes are: fanIn, fanOut`);
    }
    const std = gain / Math.sqrt(fan);
    return mx.random.normal(a.shape, dtype, null, std);
  };
  return initializer;
}

/**
 * A He uniform (Kaiming uniform) initializer.
 *
 * @remarks
 *
 * This initializer samples from a uniform distribution with a range
 * computed from the number of input (`fanIn`) or output (`fanOut`)
 * units according to:
 *
 * ```math
 * \sigma = \gamma \sqrt{\frac{3.0}{\text{fan}}}
 * ```
 *
 * where `fan` is either the number of input units when the `mode` is `"fanIn"`
 * or output units when the `mode` is `"fanOut"`.
 *
 * For more details see the original reference: [Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
 *
 * @param dtype - The data type of the array. Default: `mx.float32`.
 *
 * @returns An initializer that returns an array with the same shape as the
 * input, filled with samples from the He uniform distribution.
 *
 * @example
 * ```
 * const initFn = nn.init.heUniform();
 * initFn(mx.zeros([2, 2]));  // uses fan_in
 * // [[0.0300242, -0.0184009],
 * //  [0.793615, 0.666329]], dtype=float32)
 * initFn(mx.zeros([2, 2]), 'fanOut', 5);
 * // [[-1.64331, -2.16506],
 * //  [1.08619, 5.79854]], dtype=float32)
 * ```
 */
export function heUniform(dtype: mx.Dtype = mx.float32): (a: mx.array, mode?: 'fanIn' | 'fanOut', gain?: number) => mx.array {
  const initializer = (a: mx.array, mode: 'fanIn' | 'fanOut' = 'fanIn', gain: number = 1.0): mx.array => {
    const [fanIn, fanOut] = calculateFanInFanOut(a);
    let fan;
    if (mode === 'fanIn') {
      fan = fanIn;
    } else if (mode === 'fanOut') {
      fan = fanOut;
    } else {
      throw new Error(`Invalid mode: ${mode}. Valid modes are: fanIn, fanOut`);
    }
    const limit = gain * Math.sqrt(3.0 / fan);
    return mx.random.uniform(-limit, limit, a.shape, dtype);
  };
  return initializer;
}

/**
 * An initializer that returns a sparse matrix.
 *
 * @remarks
 *
 * Initializer return an array that has the same shape as input array with
 * the values filled with samples from a normal distribution.
 *
 * @param sparsity - The fraction of elements in each column to be set to zero.
 * @param mean - Mean of the normal distribution. Default: `0.0`.
 * @param std - Standard deviation of the normal distribution. Default: `1.0`.
 * @param dtype - The data type of the array. Default: `mx.float32`.
 *
 * @returns An initializer that returns a sparse matrix.
 *
 * @example
 * ```
 * const initFn = nn.init.sparse(0.5);
 * initFn(mx.zeros([2, 2]));
 * // [[-1.91187, -0.117483],
 * //  [0, 0]], dtype=float32)
 * ```
 */
export function sparse(sparsity: number, mean: number = 0.0, std: number = 1.0, dtype: mx.Dtype = mx.float32): (a: mx.array) => mx.array {
  const initializer = (a: mx.array): mx.array => {
    if (a.shape.length !== 2) {
      throw new Error('Only tensors with 2 dimensions are supported');
    }
    const [rows, cols] = a.shape;
    const numZeros = Math.ceil(sparsity * cols);
    const order = mx.argsort(mx.random.uniform(0, 1, a.shape), 1);
    a = mx.random.normal(a.shape, dtype, mean, std);
    a.indexPut_([mx.arange(rows, mx.int64).reshape(rows, 1), order.index(mx.Slice(), mx.Slice(0, numZeros))], 0);
    return a;
  };
  return initializer;
}

/**
 * An initializer that returns an orthogonal matrix.
 *
 * @param gain - Scaling factor for the orthogonal matrix. Default: `1.0`.
 * @param dtype - Data type of the array. Default: `float32`.
 *
 * @returns An initializer that returns an orthogonal matrix with the same shape
 * as the input.
 */
export function orthogonal(gain: number = 1.0, dtype: mx.Dtype = mx.float32): (a: mx.array) => mx.array {
  const initializer = (a: mx.array): mx.array => {
    if (a.ndim !== 2) {
      throw new Error(`Orthogonal initialization requires a 2D array but got a ${a.ndim}D array.`);
    }

    const [rows, cols] = a.shape;
    const n = Math.max(rows, cols);

    const rmat = mx.random.normal([n, n]);

    // Perform QR decomposition on CPU.
    let [q, r] = mx.linalg.qr(rmat, mx.cpu);

    // Adjust the sign of Q using the diagonal of R.
    const d = mx.diag(r);
    a = mx.multiply(q, mx.sign(d));

    // Slice Q to the desired shape
    q = q.index(mx.Slice(null, rows), mx.Slice(null, cols));

    // Scale Q by gain
    q = mx.multiply(q, gain);
    return q.astype(dtype);
  };
  return initializer;
}

// Helpers.
function calculateFanInFanOut(x: mx.array): [number, number] {
  if (x.ndim < 2) {
    throw new Error(`Glorot / He initialization requires at least 2 dimensional input but input with ${x.ndim} dimensions.`);
  }
  let fanIn = x.shape[x.shape.length - 1];
  let fanOut = x.shape[0];
  if (x.ndim > 2) {
    let receptiveField = 1;
    for (let d of x.shape.slice(1, x.shape.length - 1)) {
      receptiveField *= d;
    }
    fanIn = fanIn * receptiveField;
    fanOut = fanOut * receptiveField;
  }
  return [fanIn, fanOut];
}
