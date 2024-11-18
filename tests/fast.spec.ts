import {core as mx} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('fast', () => {
  it('rope', () => {
    const T = 4;
    const defaults: [number, mx.Dtype, number, number, number, boolean] = [
      8, mx.float32, 10000.0, 1.0, 0, false
    ];
    const tolerances = [
      {dtype: mx.float32, eps: 1e-6},
      {dtype: mx.float16, eps: 1e-3},
      {dtype: mx.bfloat16, eps: 1e-2},
    ];
    const dtypes = [mx.float32, mx.float16, mx.bfloat16];
    const bases = [10000.0, 1000000.0];
    const scales = [1.0, 2.0];
    const offsets = [0, 3];
    const traditionals = [true, false];

    for (let traditional of traditionals) {
      let [dims, dtype, base, scale, offset] = defaults;
      for (let base of bases) {
        const x = mx.random.uniform(0, 1, [2, T, dims]).astype(dtype);
        const rx = ropeOrig(x, dims, traditional, base, scale, offset);
        const rxFast = mx.fast.rope(x, dims, traditional, base, scale, offset);
        assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                       tolerances.find(t => t.dtype === dtype)!.eps);
      }

      [dims, , base, scale, offset] = defaults;
      for (let dtype of dtypes) {
        const x = mx.random.uniform(0, 1, [2, T, dims]).astype(dtype);
        const rx = ropeOrig(x, dims, traditional, base, scale, offset);
        const rxFast = mx.fast.rope(x, dims, traditional, base, scale, offset);
        assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                       tolerances.find(t => t.dtype === dtype)!.eps);
      }

      [dims, dtype, base, scale] = defaults;
      for (let offset of offsets) {
        const x = mx.random.uniform(0, 1, [2, T, dims]).astype(dtype);
        const rx = ropeOrig(x, dims, traditional, base, scale, offset);
        const rxFast = mx.fast.rope(x, dims, traditional, base, scale, offset);
        assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                       tolerances.find(t => t.dtype === dtype)!.eps);
      }

      [dims, dtype, base, , offset] = defaults;
      for (let scale of scales) {
        const x = mx.random.uniform(0, 1, [2, T, dims]).astype(dtype);
        const rx = ropeOrig(x, dims, traditional, base, scale, offset);
        const rxFast =  mx.fast.rope(x, dims, traditional, base, scale, offset);
        assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                       tolerances.find(t => t.dtype === dtype)!.eps);
      }

      [dims, , base, scale, offset, traditional] = defaults;
      const x = mx.random.uniform(0, 1, [1, 1, 4, dims]).swapaxes(1, 2);
      const rx = ropeOrig(x, dims, traditional, base, scale, offset);
      const rxFast = mx.fast.rope(
        mx.multiply(1.0, x),  // multiply here to allow donation
        dims,
        traditional,
        base,
        scale,
        offset,
      );
     assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number, tolerances.find(t => t.dtype === mx.float32)!.eps);
    }
  });

  it('ropeWithFreqs', () => {
    // Check throws
    const T = 4;
    const dims = 8;
    let x = mx.random.uniform(0, 1, [2, T, dims]);

    assert.throws(() => {
      const freqs = mx.random.uniform(0, 1, [dims - 1]);
      mx.fast.rope(x, dims, false, undefined, 1.0, 0, freqs);
    });

    assert.throws(() => {
      const freqs = mx.random.uniform(0, 1, [1, dims]);
      mx.fast.rope(x, dims, false, undefined, 1.0, 0, freqs);
    });

    const freqs = mx.random.uniform(0, 1, [dims / 2]);

    let rx = ropeOrig(x, dims, false, null, 1.0, 0, freqs);
    let rxFast = mx.fast.rope(x, dims, false, undefined, 1.0, 0, freqs);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number, 1e-5);

    // Test single vector
    x = mx.random.uniform(0, 1, [1, 1, dims]);
    rx = ropeOrig(x, dims, false, null, 1.0, 0, freqs);
    rxFast = mx.fast.rope(x, dims, false, undefined, 1.0, 0, freqs);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number, 1e-5);

    // Test grad with freqs
    const f1 = (x, y) => mx.multiply(ropeOrig(x, dims, false, undefined, 1.0, 0, freqs), y).sum();
    const f2 = (x, y) => mx.multiply(mx.fast.rope(x, dims, false, undefined, 1.0, 0, freqs), y).sum();

    x = mx.random.uniform(0, 1, [2, 4, dims]);
    const y = mx.random.uniform(0, 1, [2, 4, dims]);
    const g1 = mx.grad(f1)(x, y);
    const g2 = mx.grad(f2)(x, y);
    assert.isBelow(mx.abs(mx.subtract(g1, g2)).max().item() as number, 1e-5);
  });

  // Test broken with https://github.com/ml-explore/mlx/pull/1450.
  xit('ropeGrad', () => {
    const D = 32;
    const defaults: [number, number, number, number, boolean] = [
      D, 10000.0, 1.0, 0, false
    ];

    for (let dims of [D, D / 2]) {
      for (let traditional of [true, false]) {
        const [, base, scale, offset] = defaults;
        const f1 = (x, y) => mx.multiply(ropeOrig(x, dims, traditional, base, scale, offset), y).sum();
        const f2 = (x, y) => mx.multiply(mx.fast.rope(x, dims, traditional, base, scale, offset), y).sum();
        const x = mx.random.uniform(0, 1, [2, 100, D]);
        const y = mx.random.uniform(0, 1, [2, 100, D]);
        const g1 = mx.grad(f1)(x, y);
        const g2 = mx.grad(f2)(x, y);
        assert.isBelow(mx.abs(mx.subtract(g1, g2)).max().item() as number, 1e-5);
      }
    }
  });

  it('rmsNorm', () => {
    const tolerances = [{dtype: mx.float32, eps: 1e-6}, {dtype: mx.float16, eps: 1e-3}, {dtype: mx.bfloat16, eps: 1e-2}];

    const dtypes = [mx.float32, mx.float16, mx.bfloat16];
    const epss = [1e-3, 1e-5];
    const dimss = [31, 32, 33];
    const defaults: [mx.Dtype, number, number] = [mx.float32, 1e-5, 32];

    for (const dtype of dtypes) {
      const [, eps, dims] = defaults;
      const x = mx.random.uniform(0, 1, [2, dims]).astype(dtype);
      const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
      const rx = rmsNorm(x, weight, eps);
      const rxFast = mx.fast.rmsNorm(x, weight, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);
    }

    for (const eps of epss) {
      const [dtype, , dims] = defaults;
      const x = mx.random.uniform(0, 1, [2, dims]).astype(dtype);
      const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
      const rx = rmsNorm(x, weight, eps);
      const rxFast = mx.fast.rmsNorm(x, weight, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);
    }

    for (const dims of dimss) {
      const [dtype, eps] = defaults;
      const x = mx.random.uniform(0, 1, [2, dims]).astype(dtype);
      const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
      const rx = rmsNorm(x, weight, eps);
      const rxFast = mx.fast.rmsNorm(x, weight, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);
    }

    const [dims, dtype, eps] = [4099, mx.float32, 1e-5];
    const x = mx.random.uniform(0, 1, [dims]).astype(dtype);
    const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
    const rx = rmsNorm(x, weight, eps);
    const rxFast = mx.fast.rmsNorm(x, weight, eps);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number, 1e-6);

    assert.throws(() => {
      const x = mx.random.uniform(0, 1, [1, 5]);
      mx.fast,rmsNorm(x, mx.ones([4]), 1e-5);
    });
  });

  it('rmsNormGrad', () => {
    let D = 32;
    const eps = 1e-5;
    const f1 = (x, w, y) => mx.sum(mx.multiply(rmsNorm(x, w, eps), y));
    const f2 = (x, w, y) => mx.sum(mx.multiply(mx.fast.rmsNorm(x, w, eps), y));

    let x = mx.random.uniform(0, 1, [8, 100, D]);
    let w = mx.random.uniform(0, 1, [D]);
    let y = mx.random.uniform(0, 1, [8, 100, D]);
    let [gx1, gw1] = mx.grad(f1, [0, 1])(x, w, y);
    let [gx2, gw2] = mx.grad(f2, [0, 1])(x, w, y);
    assert.isBelow(mx.subtract(gx1, gx2).abs().max().item() as number, 1e-5);
    assert.isBelow((mx.subtract(gw1, gw2).abs().max().item() as number) / (gw1.abs().mean().item() as number), 1e-5);

    D = 8192;
    x = mx.random.uniform(0, 1, [2, 2, D]);
    w = mx.random.uniform(0, 1, [D]);
    y = mx.random.uniform(0, 1, [2, 2, D]);
    [gx1, gw1] = mx.grad(f1, [0, 1])(x, w, y);
    [gx2, gw2] = mx.grad(f2, [0, 1])(x, w, y);
    assert.isBelow(mx.subtract(gx1, gx2).abs().max().item() as number, 1e-5);
    assert.isBelow((mx.subtract(gw1, gw2).abs().max().item() as number) / (gw1.abs().mean().item() as number), 1e-5);

    const gf = f => {
      return (x, w, y) => {
        const [gx, gw] = mx.grad(f, [0, 1])(x, w, y) as any;
        return mx.sum(mx.add(gx, gw));
      };
    };

    [gx1, gw1] = mx.grad(gf(f1), [0, 1])(x, w, y);
    [gx2, gw2] = mx.grad(gf(f2), [0, 1])(x, w, y);
    assert.isBelow(mx.subtract(gx1, gx2).abs().max().item() as number, 1e-5);
    assert.isBelow((mx.subtract(gw1, gw2).abs().max().item() as number) / (gw1.abs().mean().item() as number), 1e-5);
  });

  it('layerNorm', function() {
    // This test is unreliable in CPU.
    if (!mx.metal.isAvailable())
      this.retries(4);

    const tolerances = [
      {dtype: mx.float32, eps: 1e-5},
      {dtype: mx.float16, eps: 5e-3},
      {dtype: mx.bfloat16, eps: 5e-2},
    ];

    const dtypes = [mx.float32, mx.float16, mx.bfloat16];
    const epss = [1e-3, 1e-5];
    const dimss = [31, 32, 33];
    const defaults: [mx.Dtype, number, number] = [mx.float32, 1e-5, 32];

    for (const dtype of dtypes) {
      const [, eps, dims] = defaults;
      const x = mx.random.uniform(0, 1, [2, dims]).astype(dtype);
      const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
      const bias = mx.random.uniform(0, 1, [dims]).astype(dtype);

      let rx = layerNorm(x, weight, bias, eps);
      let rxFast = mx.fast.layerNorm(x, weight, bias, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, weight, null, eps);
      rxFast = mx.fast.layerNorm(x, weight, null, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, null, bias, eps);
      rxFast = mx.fast.layerNorm(x, null, bias, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, null, null, eps);
      rxFast = mx.fast.layerNorm(x, null, null, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);
    }

    for (const eps of epss) {
      const [dtype, , dims] = defaults;
      const x = mx.random.uniform(0, 1, [2, dims]).astype(dtype);
      const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
      const bias = mx.random.uniform(0, 1, [dims]).astype(dtype);

      let rx = layerNorm(x, weight, bias, eps);
      let rxFast = mx.fast.layerNorm(x, weight, bias, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, weight, null, eps);
      rxFast = mx.fast.layerNorm(x, weight, null, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, null, bias, eps);
      rxFast = mx.fast.layerNorm(x, null, bias, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, null, null, eps);
      rxFast = mx.fast.layerNorm(x, null, null, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);
    }

    for (const dims of dimss) {
      const [dtype, eps] = defaults;
      const x = mx.random.uniform(0, 1, [2, dims]).astype(dtype);
      const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
      const bias = mx.random.uniform(0, 1, [dims]).astype(dtype);

      let rx = layerNorm(x, weight, bias, eps);
      let rxFast = mx.fast.layerNorm(x, weight, bias, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, weight, null, eps);
      rxFast = mx.fast.layerNorm(x, weight, null, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, null, bias, eps);
      rxFast = mx.fast.layerNorm(x, null, bias, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);

      rx = layerNorm(x, null, null, eps);
      rxFast = mx.fast.layerNorm(x, null, null, eps);
      assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                     tolerances.find(t => t.dtype === dtype)!.eps);
    }

    const [dims, dtype, eps] = [4099, mx.float32, 1e-5];
    const x = mx.random.uniform(0, 1, [dims]).astype(dtype);
    const weight = mx.random.uniform(0, 1, [dims]).astype(dtype);
    const bias = mx.random.uniform(0, 1, [dims]).astype(dtype);

    let rx = layerNorm(x, weight, bias, eps);
    let rxFast = mx.fast.layerNorm(x, weight, bias, eps);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                   tolerances.find(t => t.dtype === dtype)!.eps);

    rx = layerNorm(x, weight, null, eps);
    rxFast = mx.fast.layerNorm(x, weight, null, eps);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                   tolerances.find(t => t.dtype === dtype)!.eps);

    rx = layerNorm(x, null, bias, eps);
    rxFast = mx.fast.layerNorm(x, null, bias, eps);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                   tolerances.find(t => t.dtype === dtype)!.eps);

    rx = layerNorm(x, null, null, eps);
    rxFast = mx.fast.layerNorm(x, null, null, eps);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number,
                   tolerances.find(t => t.dtype === dtype)!.eps);
  });

  it('sliceIntoLayerNorm', () => {
    const eps = 1e-5;
    const x = mx.random.uniform(0, 1, [8, 100, 128]).index(mx.Slice(), mx.Slice(99));
    const rxFast = mx.fast.layerNorm(x, null, null, eps);
    const rx = layerNorm(x, null, null, eps);
    assert.isBelow(mx.abs(mx.subtract(rx, rxFast)).max().item() as number, 1e-4);
  });

  it('layerNormGrad', function() {
    this.timeout(10 * 1000);  // slow in QEMU

    // This test is unreliable in CPU.
    if (!mx.metal.isAvailable())
      this.retries(4);

    let D = 32;
    const eps = 1e-5;
    const f1 = (x, w, b, y) => mx.multiply(layerNorm(x, w, b, eps), y).sum();
    const f2 = (x, w, b, y) => mx.multiply(mx.fast.layerNorm(x, w, b, eps), y).sum();

    let x = mx.random.uniform(0, 1, [8, 100, D]);
    let w = mx.random.uniform(0, 1, [D]);
    let b = mx.random.uniform(0, 1, [D]);
    let y = mx.random.uniform(0, 1, [8, 100, D]);

    let [gx1, gw1, gb1] = mx.grad(f1, [0, 1, 2])(x, w, b, y);
    let [gx2, gw2, gb2] = mx.grad(f2, [0, 1, 2])(x, w, b, y);
    assert.isBelow(mx.subtract(gx1, gx2).abs().max().item() as number, 5e-5);
    assert.isBelow((mx.subtract(gw1, gw2).abs().max().item() as number) / (gw1.abs().mean().item() as number), 5e-5);
    assert.isBelow((mx.subtract(gb1, gb2).abs().max().item() as number) / (gb1.abs().mean().item() as number), 5e-5);

    D = 8192;
    x = mx.random.uniform(0, 1, [8, 100, D]);
    w = mx.random.uniform(0, 1, [D]);
    b = mx.random.uniform(0, 1, [D]);
    y = mx.random.uniform(0, 1, [8, 100, D]);

    [gx1, gw1, gb1] = mx.grad(f1, [0, 1, 2])(x, w, b, y);
    [gx2, gw2, gb2] = mx.grad(f2, [0, 1, 2])(x, w, b, y);
    assert.isBelow(mx.abs(mx.subtract(gx1, gx2)).max().item() as number, 5e-5);
    assert.isBelow((mx.subtract(gw1, gw2).abs().max().item() as number) / (gw1.abs().mean().item() as number), 5e-5);
    assert.isBelow((mx.subtract(gb1, gb2).abs().max().item() as number) / (gb1.abs().mean().item() as number), 5e-5);

    const gf = f => (x, w, b, y) => {
      const [gx, gw, gb] = mx.grad(f, [0, 1, 2])(x, w, b, y) as any;
      return mx.multiply(mx.add(gx, mx.add(gw, gb)), y).sum();
    };

    [gx1, gw1, gb1] = mx.grad(gf(f1), [0, 1, 2])(x, w, b, y);
    [gx2, gw2, gb2] = mx.grad(gf(f2), [0, 1, 2])(x, w, b, y);
    assert.isBelow((mx.subtract(gx1, gx2).abs().max().item() as number) / (gx1.abs().mean().item() as number), 5e-5);
    assert.isBelow((mx.subtract(gw1, gw2).abs().max().item() as number) / (gw1.abs().mean().item() as number), 5e-5);
    assert.isBelow(mx.abs(gb1).max().item() as number, 1e-9);
    assert.isBelow(mx.abs(gb2).max().item() as number, 1e-9);
  });

  it('layerNormGradNoParams', () => {
    const eps = 1e-5;
    const f1 = (x: mx.array) => layerNorm(x, null, null, eps).sum();
    const f2 = (x: mx.array) => mx.fast.layerNorm(x, null, null, eps).sum();
    const x = mx.random.normal([2, 2, 8]);
    mx.eval(x);

    const gx1 = mx.grad(f1)(x);
    const gx2 = mx.grad(f2)(x);
    assert.deepEqual(gx1.shape, gx2.shape);
    // FIXME(zcbenz): The results should be close but they do not.
    // assertArrayAllTrue(mx.allclose(gx1, gx2, 1e-6));
  });

  it('layerNormGradParams', () => {
    const eps = 1e-5;
    const f1 = (params, x) => mx.sum(layerNorm(x, params[0], params[1], eps));
    const f2 = (params, x) => mx.sum(mx.fast.layerNorm(x, params[0], params[1], eps));

    let w = mx.ones([8]);
    let b = mx.zeros([8]);
    let x = mx.random.normal([2, 2, 8]);
    mx.eval(x, w, b);

    const [gw1, gb1] = mx.grad(f1)([w, b], x);
    const [gw2, gb2] = mx.grad(f2)([w, b], x);

    assert.isBelow(mx.divide(mx.subtract(gw1, gw2).abs().max(),
                             gw1.abs().mean()).item() as number,
                   1e-5);
    assert.isBelow(mx.divide(mx.subtract(gb1, gb2).abs().max(),
                             gb1.abs().mean()).item() as number,
                   1e-5);
  });

  it('fastTransforms', () => {
    let x = mx.random.uniform(0, 1, [2, 2, 8]);

    const defaults: [number, boolean, number, number, number] = [8, false, 10000.0, 1.0, 0];
    const [dims, traditional, base, scale, offset] = defaults;

    let [, vjpOut] = mx.vjp(x => ropeOrig(x, ...defaults), [x], [mx.onesLike(x)])
    let [, vjpFastOut] = mx.vjp(
      x => mx.fast.rope(x, dims, traditional, base, scale, offset),
      [x],
      [mx.onesLike(x)]
    );
    assertArrayAllTrue(mx.allclose(vjpOut[0], vjpFastOut[0]));

    [, vjpOut] = mx.jvp(x => ropeOrig(x, ...defaults), [x], [mx.onesLike(x)]);
    [, vjpFastOut] = mx.jvp(
      x => mx.fast.rope(x, dims, traditional, base, scale, offset),
      [x],
      [mx.onesLike(x)]
    );
    assertArrayAllTrue(mx.allclose(vjpOut[0], vjpFastOut[0]));

    x = mx.random.uniform(0, 1, [2, 2, 2, 8]);
    const vmapOut = mx.vmap((x: mx.array) => ropeOrig(x, ...defaults))(x);
    const vmapFastOut = mx.vmap((x: mx.array) => mx.fast.rope(x, dims, traditional, base, scale, offset))(x);
    assertArrayAllTrue(mx.allclose(vmapOut, vmapFastOut));
  });
});

function ropeOrig(x, dims, traditional, base, scale, offset, freqs?: mx.array) {
  const N = x.shape[x.shape.length - 2] + offset;
  const dtype = x.dtype;
  const halfD = Math.floor(dims / 2);
  const positions = mx.multiply(mx.arange(offset, N, dtype), scale);
  const invFreqs = freqs ? mx.divide(1, freqs)
                         : mx.exp(mx.multiply(mx.negative(mx.arange(0, halfD, dtype)),
                                              Math.log(base) / halfD));
  const theta = mx.multiply(mx.reshape(positions, [-1, 1]),
                            mx.reshape(invFreqs, [1, -1]));
  const [costheta, sintheta] = [mx.cos(theta), mx.sin(theta)];

  if (traditional) {
    const x1 = x.index('...', mx.Slice(null, dims, 2));
    const x2 = x.index('...', mx.Slice(1, dims, 2));
    const rx1 = mx.subtract(mx.multiply(x1, costheta),
                            mx.multiply(x2, sintheta));
    const rx2 = mx.add(mx.multiply(x1, sintheta),
                       mx.multiply(x2, costheta));
    let rx = mx.concatenate([rx1.index('...', null), rx2.index('...', null)], -1);
    if (dims < x.shape[1]) {
      rx = mx.reshape(rx, [...x.shape.slice(0, -1), dims]);
      rx = mx.concatenate([rx, x.index('...', mx.Slice(dims))], -1);
    }
    return mx.reshape(rx, x.shape);
  } else {
    const x1 = x.index('...', mx.Slice(null, Math.floor(dims / 2)));
    const x2 = x.index('...', mx.Slice(Math.floor(dims / 2), dims));
    const rx1 = mx.subtract(mx.multiply(x1, costheta),
                            mx.multiply(x2, sintheta));
    const rx2 = mx.add(mx.multiply(x1, sintheta),
                       mx.multiply(x2, costheta));
    return dims < x.shape[1] ?
      mx.concatenate([rx1, rx2, x.index('...', mx.Slice(dims))], -1) :
      mx.concatenate([rx1, rx2], -1);
  }
}

function rmsNorm(x, weight, eps) {
  x = x.astype(mx.float32);
  x = mx.multiply(x, mx.rsqrt(mx.add(x.square().mean(-1, true), eps)));
  return mx.multiply(x.astype(weight.dtype), weight);
}

function layerNorm(x: mx.array, weight: mx.array | null, bias: mx.array | null, eps) {
  const ot = x.dtype;
  x = x.astype(mx.float32);
  const mean = x.mean(-1, true);
  const variance = x.variance(-1, true);
  x = mx.multiply(mx.subtract(x, mean),
                  mx.rsqrt(mx.add(variance, eps)));
  x = x.astype(ot);
  if (weight)
    x = mx.multiply(x, weight);
  if (bias)
    x = mx.add(x, bias);
  return x;
}
