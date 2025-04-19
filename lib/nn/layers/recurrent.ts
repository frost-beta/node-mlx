import {core as mx} from '../../core';
import {tanh} from './activations';
import {Module} from './base';

/**
 * An Elman recurrent layer.
 *
 * @remarks
 *
 * The input is a sequence of shape `NLD` or `LD` where:
 * * `N` is the optional batch dimension
 * * `L` is the sequence length
 * * `D` is the input's feature dimension
 *
 * Concretely, for each element along the sequence length axis, this layer
 * applies the function:
 *
 * ```math
 * h_{t + 1} = \text{tanh} (W_{ih}x_t + W_{hh}h_t + b)
 * ```
 *
 * The hidden state `h` has shape `NH` or `H`, depending on whether the input is
 * batched or not. Returns the hidden state at each time step, of shape `NLH` or
 * `LH`.
 *
 * @param inputDims - Dimension of the input, `D`.
 * @param hiddenDims - Dimension of the hidden state, `H`.
 * @param bias - Whether to use a bias. Default: `true`.
 * @param nonlinearity - Non-linearity to use. If `null`, then func:`tanh` is
 * used. Default: `null`.
 */
export class RNN extends Module {
  nonlinearity: (x: mx.array) => mx.array;
  hiddenDims: number;
  Wxh: mx.array;
  Whh: mx.array;
  bias?: mx.array;

  constructor(inputDims: number,
              hiddenDims: number,
              bias = true,
              nonlinearity: (x: mx.array) => mx.array | null = null) {
    super();
    this.nonlinearity = nonlinearity ?? tanh;
    if (typeof this.nonlinearity !== 'function') {
      throw Error(`Nonlinearity must be callable. Current value: ${nonlinearity}`);
    }

    const scale = 1.0 / Math.sqrt(hiddenDims);
    this.hiddenDims = hiddenDims;
    this.Wxh = mx.random.uniform(-scale, scale, [hiddenDims, inputDims]);
    this.Whh = mx.random.uniform(-scale, scale, [hiddenDims, hiddenDims]);
    if (bias) {
      this.bias = mx.random.uniform(-scale, scale, [hiddenDims]);
    }
  }

  override toStringExtra(): string {
    return `inputDims=${this.Wxh.shape[1]}, hiddenDims=${this.hiddenDims}, ` +
           `nonlinearity=${this.nonlinearity}, bias=${!!this.bias}`;
  }

  override forward(x: mx.array, hidden?: mx.array): mx.array {
    if (this.bias)
      x = mx.addmm(this.bias, x, this.Wxh.T);
    else
      x = mx.matmul(x, this.Wxh.T);

    const allHidden = [];
    for (let i = 0; i < x.shape.at(-2); i++) {
      if (hidden) {
        hidden = mx.addmm(x.index('...', i, mx.Slice()), hidden, this.Whh.T);
      } else {
        hidden = x.index('...', i, mx.Slice());
      }
      hidden = this.nonlinearity(hidden);
      allHidden.push(hidden);
    }

    return mx.stack(allHidden, -2);
  }
}

/**
 * A gated recurrent unit (GRU) RNN layer.
 *
 * @remarks
 *
 * The input has shape `NLD` or `LD` where:
 * * `N` is the optional batch dimension
 * * `L` is the sequence length
 * * `D` is the input's feature dimension
 *
 * Concretely, for each element of the sequence, this layer computes:
 *
 * ```math
 * \begin{aligned}
 * r_t &= \sigma (W_{xr}x_t + W_{hr}h_t + b_{r}) \\
 * z_t &= \sigma (W_{xz}x_t + W_{hz}h_t + b_{z}) \\
 * n_t &= \text{tanh}(W_{xn}x_t + b_{n} + r_t \odot (W_{hn}h_t + b_{hn})) \\
 * h_{t + 1} &= (1 - z_t) \odot n_t + z_t \odot h_t
 * \end{aligned}
 * ```
 *
 * The hidden state `h` has shape `NH` or `H` depending on whether the input is
 * batched or not. Returns the hidden state at each time step of shape `NLH` or
 * `LH`.
 *
 * @param inputDims - Dimension of the input, `D`.
 * @param hiddenDims - Dimension of the hidden state, `H`.
 * @param bias - Whether to use biases or not. Default: `true`.
 */
export class GRU extends Module {
  hiddenDims: number;
  Wx: mx.array;
  Wh: mx.array;
  b?: mx.array;
  bhn?: mx.array;

  constructor(inputDims: number, hiddenDims: number, bias = true) {
    super();
    this.hiddenDims = hiddenDims;

    const scale = 1.0 / Math.sqrt(hiddenDims);
    this.Wx = mx.random.uniform(-scale, scale, [3 * hiddenDims, inputDims]);
    this.Wh = mx.random.uniform(-scale, scale, [3 * hiddenDims, hiddenDims]);
    if (bias) {
      this.b = mx.random.uniform(-scale, scale, [3 * hiddenDims]);
      this.bhn = mx.random.uniform(-scale, scale, [hiddenDims]);
    }
  }

  override toStringExtra(): string {
    return `inputDims=${this.Wx.shape[1]}, hiddenDims=${this.hiddenDims}, bias=${!!this.b}`;
  }

  override forward(x: mx.array, hidden?: mx.array): mx.array {
    if (this.b)
      x = mx.addmm(this.b, x, this.Wx.T);
    else
      x = mx.matmul(x, this.Wx.T);

    const xRz = x.index('...', mx.Slice(null, -this.hiddenDims));
    const xN = x.index('...', mx.Slice(-this.hiddenDims));

    const allHidden = [];
    for (let i = 0; i < x.shape[x.shape.length - 2]; i++) {
      let rz = xRz.index('...', i, mx.Slice());
      let hProjN;
      if (hidden) {
        const hProj = mx.matmul(hidden, this.Wh.T);
        const hProjRz = hProj.index('...', mx.Slice(null, -this.hiddenDims));
        hProjN = hProj.index('...', mx.Slice(-this.hiddenDims));
        if (this.bhn)
          hProjN = mx.add(hProjN, this.bhn);
        rz = mx.add(rz, hProjRz);
      }

      rz = mx.sigmoid(rz);
      const [r, z] = mx.split(rz, 2, -1);

      let n = xN.index('...', i, mx.Slice());
      if (hProjN)
        n = mx.add(n, mx.multiply(r, hProjN));
      n = mx.tanh(n);

      if (hidden) {
        hidden = mx.add(mx.multiply(mx.subtract(1, z), n),
                        mx.multiply(z, hidden));
      } else {
        hidden = mx.multiply(mx.subtract(1, z), n);
      }

      allHidden.push(hidden);
    }

    return mx.stack(allHidden, -2);
  }
}

/**
 * An LSTM recurrent layer.
 *
 * @remarks
 *
 * The input has shape `NLD` or `LD` where:
 *
 * * `N` is the optional batch dimension
 * * `L` is the sequence length
 * * `D` is the input's feature dimension
 *
 * Concretely, for each element of the sequence, this layer computes:
 *
 * ```math
 * \begin{aligned}
 * i_t &= \sigma (W_{xi}x_t + W_{hi}h_t + b_{i}) \\
 * f_t &= \sigma (W_{xf}x_t + W_{hf}h_t + b_{f}) \\
 * g_t &= \text{tanh} (W_{xg}x_t + W_{hg}h_t + b_{g}) \\
 * o_t &= \sigma (W_{xo}x_t + W_{ho}h_t + b_{o}) \\
 * c_{t + 1} &= f_t \odot c_t + i_t \odot g_t \\
 * h_{t + 1} &= o_t \text{tanh}(c_{t + 1})
 * \end{aligned}
 * ```
 *
 * The hidden state `h` and cell state `c` have shape `NH` or `H`, depending on
 * whether the input is batched or not.
 *
 * The layer returns two arrays, the hidden state and the cell state at each
 * time step, both of shape `NLH` or `LH`.
 *
 * @param inputDims - Dimension of the input, `D`.
 * @param hiddenDims - Dimension of the hidden state, `H`.
 * @param bias - Whether to use biases or not. Default: `true`.
 */
export class LSTM extends Module {
  hiddenDims: number;
  Wx: mx.array;
  Wh: mx.array;
  bias?: mx.array;

  constructor(inputDims: number, hiddenDims: number, bias = true) {
    super();
    this.hiddenDims = hiddenDims;

    const scale = 1.0 / Math.sqrt(hiddenDims);
    this.Wx = mx.random.uniform(-scale, scale, [4 * hiddenDims, inputDims]);
    this.Wh = mx.random.uniform(-scale, scale, [4 * hiddenDims, hiddenDims]);
    if (bias) {
      this.bias = mx.random.uniform(-scale, scale, [4 * hiddenDims]);
    }
  }

  override toStringExtra(): string {
    return `inputDims=${this.Wx.shape[1]}, hiddenDims=${this.hiddenDims}, bias=${!!this.bias}`;
  }

  override forward(x: mx.array, hidden?: mx.array, cell?: mx.array): [mx.array, mx.array] {
    if (this.bias)
      x = mx.addmm(this.bias, x, this.Wx.T);
    else
      x = mx.matmul(x, this.Wx.T);

    const allHidden: mx.array[] = [];
    const allCell: mx.array[] = [];
    for (let j = 0; j < x.shape[x.shape.length - 2]; j++) {
      let ifgo = x.index('...', j, mx.Slice());
      if (hidden)
        ifgo = mx.addmm(ifgo, hidden, this.Wh.T);
      const [si, sf, sg, so] = mx.split(ifgo, 4, -1);

      const i = mx.sigmoid(si);
      const f = mx.sigmoid(sf);
      const g = mx.tanh(sg);
      const o = mx.sigmoid(so);

      if (cell)
        cell = mx.add(mx.multiply(f, cell), mx.multiply(i, g));
      else
        cell = mx.multiply(i, g);
      hidden = mx.multiply(o, mx.tanh(cell));

      allCell.push(cell);
      allHidden.push(hidden);
    }

    return [mx.stack(allHidden, -2), mx.stack(allCell, -2)];
  }
}
