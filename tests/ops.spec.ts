import mx from '..';
import tf from '@tensorflow/tfjs';
import {assertArrayAllTrue, assertArrayAllFalse} from './utils';
import {assert} from 'chai';

describe('ops', () => {
  it('fullOnesZeros', () => {
    let x = mx.full(2, 3.0);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(x.tolist(), [3.0, 3.0]);

    x = mx.full([2, 3], 2.0);
    assert.equal(x.dtype, mx.float32);
    assert.deepEqual(x.shape, [2, 3]);
    assert.deepEqual(x.tolist(), [[2, 2, 2], [2, 2, 2]]);

    x = mx.full([3, 2], mx.array([false, true]));
    assert.equal(x.dtype, mx.bool);
    assert.deepEqual(x.tolist(), [[false, true], [false, true], [false, true]]);

    x = mx.full([3, 2], mx.array([2.0, 3.0]));
    assert.deepEqual(x.tolist(), [[2, 3], [2, 3], [2, 3]]);

    x = mx.zeros(2);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(x.tolist(), [0.0, 0.0]);

    x = mx.ones(2);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(x.tolist(), [1.0, 1.0]);

    const types = [mx.bool, mx.int32, mx.float32];
    for (const t of types) {
      x = mx.zeros([2, 2], t);
      assert.equal(x.dtype, t);
      assertArrayAllTrue(mx.equal(x, mx.array([[0, 0], [0, 0]])));
      let y = mx.zerosLike(x);
      assert.equal(y.dtype, t);
      assertArrayAllTrue(mx.equal(y, x));

      x = mx.ones([2, 2], t);
      assert.equal(x.dtype, t);
      assertArrayAllTrue(mx.equal(x, mx.array([[1, 1], [1, 1]])));
      y = mx.onesLike(x);
      assert.equal(y.dtype, t);
      assertArrayAllTrue(mx.equal(y, x));
    }
  });

  it('scalarInputs', () => {
    let a = mx.add(false, true);
    assert.equal(a.dtype, mx.bool);
    assert.equal(a.item(), true);

    a = mx.add(1, 2);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3);

    a = mx.add(1.0, 2.0);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3.0);

    a = mx.add(true, 2);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3);

    a = mx.add(true, 2.0);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3.0);

    a = mx.add(1, 2.0);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3.0);

    a = mx.add(2, true);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3);

    a = mx.add(2.0, true);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3.0);

    a = mx.add(2.0, 1);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 3.0);

    a = mx.add(mx.array(true), false);
    assert.equal(a.dtype, mx.bool);
    assert.equal(a.item(), true);

    a = mx.add(mx.array(1), false);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 1.0);

    a = mx.add(mx.array(true), 1);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 2);

    a = mx.add(mx.array(1.0), 1);
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 2.0);

    a = mx.add(1, mx.array(1.0));
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.item(), 2.0);
  });

  it('add', () => {
    let x = mx.array(1);
    let y = mx.array(1);
    let z = mx.add(x, y);
    assert.equal(z.item(), 2);

    x = mx.array(false, mx.bool);
    z = mx.add(x, 1);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 1);
    z = mx.add(2, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 2);

    x = mx.array(1, mx.uint32);
    z = mx.add(x, 3);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4);

    z = mx.add(3, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4);

    z = mx.add(x, 3.0);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4.0);

    z = mx.add(3.0, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4.0);

    x = mx.array(1, mx.int64);
    z = mx.add(x, 3);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4);
    z = mx.add(3, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4);
    z = mx.add(x, 3.0);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4.0);
    z = mx.add(3.0, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4.0);

    x = mx.array(1, mx.float32);
    z = mx.add(x, 3);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4);
    z = mx.add(3, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 4);
  });

  it('subtract', () => {
    let x = mx.array(4.0);
    let y = mx.array(3.0);

    let z = mx.subtract(x, y);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 1.0);

    z = mx.subtract(x, 3.0);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 1.0);

    z = mx.subtract(5.0, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 1.0);
  });

  it('multiply', () => {
    let x = mx.array(2.0);
    let y = mx.array(3.0);

    let z = mx.multiply(x, y);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 6.0);

    z = mx.multiply(x, 3.0);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 6.0);

    z = mx.multiply(3.0, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 6.0);
  });

  it('divide', () => {
    let x = mx.array(2.0);
    let y = mx.array(4.0);

    let z = mx.divide(x, y);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 0.5);

    z = mx.divide(x, 4.0);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 0.5);

    z = mx.divide(1.0, x);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 0.5);

    x = x.astype(mx.float16);
    z = mx.divide(x, 4.0);
    assert.equal(z.dtype, mx.float32);

    x = x.astype(mx.float16);
    z = mx.divide(4.0, x);
    assert.equal(z.dtype, mx.float32);

    x = mx.array(5);
    y = mx.array(2);
    z = mx.divide(x, y);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 2.5);

    z = mx.floorDivide(x, y);
    assert.equal(z.dtype, mx.float32);
    assert.equal(z.item(), 2);
  });

  it('remainder', () => {
    const types = [mx.float32];
    for (const dt of types) {
      let x = mx.array(2, dt);
      let y = mx.array(4, dt);

      let z1 = mx.remainder(x, y);
      let z2 = mx.remainder(y, x);
      assert.equal(z1.dtype, dt);
      assert.equal(z1.item(), 2);
      assert.equal(z2.item(), 0);

      let z = mx.remainder(x, 4);
      assert.equal(z.dtype, dt);
      assert.equal(z.item(), 2);

      z = mx.remainder(1, x);
      assert.equal(z.dtype, dt);
      assert.equal(z.item(), 1);

      z = mx.remainder(-1, x);
      assert.equal(z.dtype, dt);
      assert.equal(z.item(), 1);

      z = mx.remainder(-1, mx.negative(x));
      assert.equal(z.dtype, dt);
      assert.equal(z.item(), -1);

      x = mx.subtract(mx.arange(10).astype(dt), 5);
      y = mx.remainder(x, 5);
      z = mx.remainder(x, -5);
      assert.deepEqual(y.tolist(), [-0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
      assert.deepEqual(z.tolist(), [-0, -4, -3, -2, -1, 0, -4, -3, -2, -1]);
    }
  });

  it('comparisons', () => {
    let a = mx.array([0.0, 1.0, 5.0]);
    let b = mx.array([-1.0, 2.0, 5.0]);

    assert.deepEqual(mx.less(a, b).tolist(), [false, true, false]);
    assert.deepEqual(mx.lessEqual(a, b).tolist(), [false, true, true]);
    assert.deepEqual(mx.greater(a, b).tolist(), [true, false, false]);
    assert.deepEqual(mx.greaterEqual(a, b).tolist(), [true, false, true]);

    assert.deepEqual(mx.less(a, 5).tolist(), [true, true, false]);
    assert.deepEqual(mx.less(5, a).tolist(), [false, false, false]);
    assert.deepEqual(mx.lessEqual(5, a).tolist(), [false, false, true]);
    assert.deepEqual(mx.greater(a, 1).tolist(), [false, false, true]);
    assert.deepEqual(mx.greaterEqual(a, 1).tolist(), [false, true, true]);

    a = mx.array([0.0, 1.0, 5.0, -1.0]);
    b = mx.array([0.0, 2.0, 5.0, 3.0]);
    assert.deepEqual(mx.equal(a, b).tolist(), [true, false, true, false]);
    assert.deepEqual(mx.notEqual(a, b).tolist(), [false, true, false, true]);
  });

  it('arrayEqual', () => {
    let x = mx.array([1, 2, 3, 4]);
    let y = mx.array([1, 2, 3, 4]);
    assertArrayAllTrue(mx.arrayEqual(x, y));

    y = mx.array([1, 2, 4, 5]);
    assertArrayAllFalse(mx.arrayEqual(x, y));

    y = mx.array([1, 2, 3]);
    assertArrayAllFalse(mx.arrayEqual(x, y));

    y = mx.array([1.0, 2.0, 3.0, 4.0]);
    assertArrayAllTrue(mx.arrayEqual(x, y));

    x = mx.array([0.0, NaN]);
    y = mx.array([0.0, NaN]);
    assertArrayAllFalse(mx.arrayEqual(x, y));

    const types = [mx.float32, mx.float16, mx.bfloat16, mx.complex64];
    for (const t of types) {
      x = mx.array([0.0, NaN]).astype(t);
      y = mx.array([0.0, NaN]).astype(t);
      assertArrayAllFalse(mx.arrayEqual(x, y));
    }
  });

  it('isnan', () => {
    let x = mx.array([0.0, NaN]);
    assert.deepEqual(mx.isnan(x).tolist(), [false, true]);

    x = mx.array([0.0, NaN]).astype(mx.float16);
    assert.deepEqual(mx.isnan(x).tolist(), [false, true]);

    x = mx.array([0.0, NaN]).astype(mx.bfloat16);
    assert.deepEqual(mx.isnan(x).tolist(), [false, true]);

    x = mx.array([0.0, NaN]).astype(mx.complex64);
    assert.deepEqual(mx.isnan(x).tolist(), [false, true]);

    assert.equal(mx.isnan(mx.multiply(0, Infinity)).tolist(), true);
  });

  it('isinf', () => {
    let x = mx.array([0.0, Infinity]);
    assert.deepEqual(mx.isinf(x).tolist(), [false, true]);

    x = mx.array([0.0, Infinity]).astype(mx.float16);
    assert.deepEqual(mx.isinf(x).tolist(), [false, true]);

    x = mx.array([0.0, Infinity]).astype(mx.bfloat16);
    assert.deepEqual(mx.isinf(x).tolist(), [false, true]);

    assert.equal(mx.isinf(mx.multiply(0, Infinity)).tolist(), false);

    x = mx.array([-2147483648, 0, 2147483647], mx.int32);
    let result = mx.isinf(x);
    assert.deepEqual(result.tolist(), [false, false, false]);

    x = mx.array([-32768, 0, 32767], mx.int16);
    result = mx.isinf(x);
    assert.deepEqual(result.tolist(), [false, false, false]);
  });

  it('minimum', () => {
    const x = mx.array([0.0, -5, 10.0]);
    const y = mx.array([1.0, -7.0, 3.0]);
    const expected = [0, -7, 3];
    assert.deepEqual(mx.minimum(x, y).tolist(), expected);

    const a = mx.array([NaN]);
    const b = mx.array([0.0]);
    assert.isTrue(isNaN(mx.minimum(a, b).item() as number));
    assert.isTrue(isNaN(mx.minimum(b, a).item() as number));
  });

  it('maximum', () => {
    const x = mx.array([0.0, -5, 10.0]);
    const y = mx.array([1.0, -7.0, 3.0]);
    const expected = [1, -5, 10];
    assert.deepEqual(mx.maximum(x, y).tolist(), expected);

    const a = mx.array([NaN]);
    const b = mx.array([0.0]);
    assert.isTrue(isNaN(mx.maximum(a, b).item() as number));
    assert.isTrue(isNaN(mx.maximum(b, a).item() as number));
  });

  it('floor', () => {
    const x = mx.array([-22.03, 19.98, -27, 9, 0.0, -Infinity, Infinity]);
    const expected = [-23, 19, -27, 9, 0, -Infinity, Infinity];
    assert.deepEqual(mx.floor(x).tolist(), expected);
  });

  it('ceil', () => {
    const x = mx.array([-22.03, 19.98, -27, 9, 0.0, -Infinity, Infinity]);
    const expected = [-22, 20, -27, 9, 0, -Infinity, Infinity];
    assert.deepEqual(mx.ceil(x).tolist(), expected);
  });

  it('isposinf', () => {
    let x = mx.array([0.0, Number.NEGATIVE_INFINITY]);
    assert.deepEqual(mx.isposinf(x).tolist(), [false, false]);

    x = mx.array([0.0, Number.NEGATIVE_INFINITY]).astype(mx.float16);
    assert.deepEqual(mx.isposinf(x).tolist(), [false, false]);

    x = mx.array([0.0, Number.NEGATIVE_INFINITY]).astype(mx.bfloat16);
    assert.deepEqual(mx.isposinf(x).tolist(), [false, false]);

    x = mx.array([0.0, Number.NEGATIVE_INFINITY]).astype(mx.complex64);
    assert.deepEqual(mx.isposinf(x).tolist(), [false, false]);

    assert.equal(mx.isposinf(mx.multiply(0, Number.POSITIVE_INFINITY)).tolist(), false);

    x = mx.array([-2147483648, 0, 2147483647], mx.int32);
    let result = mx.isposinf(x);
    assert.deepEqual(result.tolist(), [false, false, false]);

    x = mx.array([-32768, 0, 32767], mx.int16);
    result = mx.isposinf(x);
    assert.deepEqual(result.tolist(), [false, false, false]);
  });

  it('isneginf', () => {
    let x = mx.array([0.0, Number.NEGATIVE_INFINITY]);
    assert.deepEqual(mx.isneginf(x).tolist(), [false, true]);

    x = mx.array([0.0, Number.NEGATIVE_INFINITY]).astype(mx.float16);
    assert.deepEqual(mx.isneginf(x).tolist(), [false, true]);

    x = mx.array([0.0, Number.NEGATIVE_INFINITY]).astype(mx.bfloat16);
    assert.deepEqual(mx.isneginf(x).tolist(), [false, true]);

    x = mx.array([0.0, Number.NEGATIVE_INFINITY]).astype(mx.complex64);
    assert.deepEqual(mx.isneginf(x).tolist(), [false, true]);

    assert.equal(mx.isneginf(mx.multiply(0, Number.POSITIVE_INFINITY)).tolist(), false);

    x = mx.array([-2147483648, 0, 2147483647], mx.int32);
    let result = mx.isneginf(x);
    assert.deepEqual(result.tolist(), [false, false, false]);

    x = mx.array([-32768, 0, 32767], mx.int16);
    result = mx.isneginf(x);
    assert.deepEqual(result.tolist(), [false, false, false]);
  });

  it('round', () => {
    let x = mx.array([0.5, -0.5, 1.5, -1.5, -21.03, 19.98, -27, 9, 0.0, -Infinity, Infinity]);
    let expected = [0, -0, 2, -2, -21, 20, -27, 9, 0, -Infinity, Infinity];
    assert.deepEqual(mx.round(x).tolist(), expected);

    let y0 = mx.round(mx.array([15, 122], mx.int32), 0);
    let y1 = mx.round(mx.array([15, 122], mx.int32), -1);
    let y2 = mx.round(mx.array([15, 122], mx.int32), -2);
    assert.equal(y0.dtype, mx.int32);
    assert.equal(y1.dtype, mx.int32);
    assert.equal(y2.dtype, mx.int32);
    assert.deepEqual(y0.tolist(), [15, 122]);
    assert.deepEqual(y1.tolist(), [20, 120]);
    assert.deepEqual(y2.tolist(), [0, 100]);

    y1 = mx.round(mx.array([1.537, 1.471], mx.float32), 1);
    y2 = mx.round(mx.array([1.537, 1.471], mx.float32), 2);
    assertArrayAllTrue(mx.allclose(y1, mx.array([1.5, 1.5])));
    assertArrayAllTrue(mx.allclose(y2, mx.array([1.54, 1.47])));

    const dtypes = [mx.bfloat16, mx.float16, mx.float32];
    for (const dtype of dtypes) {
      x = mx.subtract(mx.arange(10, dtype), 4.5);
      x = mx.round(x);
      assert.deepEqual(
        x.astype(mx.float32).tolist(),
        [-4.0, -4.0, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, 4.0]
      );
    }
  });

  it('transposeNoargs', () => {
    const x = mx.array([[0, 1, 1], [1, 0, 0]]);
    const expected = [
      [0, 1],
      [1, 0],
      [1, 0],
    ];
    assert.deepEqual(mx.transpose(x).tolist(), expected);
  });

  it('transposeAxis', () => {
    const x = mx.array([
      [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
      [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    ]);
    const expected = [
      [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
      [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]]
    ];
    assert.deepEqual(mx.transpose(x, [0, 2, 1]).tolist(), expected);
  });

  it('moveSwapAxes', () => {
    let x = mx.zeros([2, 3, 4]);
    assert.deepEqual(mx.moveaxis(x, 0, 2).shape, [3, 4, 2]);
    assert.deepEqual(x.moveaxis(0, 2).shape, [3, 4, 2]);
    assert.deepEqual(mx.swapaxes(x, 0, 2).shape, [4, 3, 2]);
    assert.deepEqual(x.swapaxes(0, 2).shape, [4, 3, 2]);
  });

  it('sum', () => {
    const x = mx.array([[1, 2],[3, 3]]);
    assert.equal(mx.sum(x).item(), 9);
    let y = mx.sum(x, true);
    assertArrayAllTrue(mx.equal(y, mx.array(9)));
    assert.deepEqual(y.shape, [1, 1]);
    assert.deepEqual(mx.sum(x, 0).tolist(), [4, 5]);
    assert.deepEqual(mx.sum(x, 1).tolist(), [3, 6]);
  });

  it('prod', () => {
    const x = mx.array([[1, 2],[3, 3]]);
    assert.equal(mx.prod(x).item(), 18);
    let y = mx.prod(x, true);
    assertArrayAllTrue(mx.equal(y, mx.array(18)));
    assert.deepEqual(y.shape, [1, 1]);
    assert.deepEqual(mx.prod(x, 0).tolist(), [3, 6]);
    assert.deepEqual(mx.prod(x, 1).tolist(), [2, 9]);
  });

  it('minAndMax', () => {
    const x = mx.array([[1, 2],[3, 4]]);
    assert.equal(mx.min(x).item(), 1);
    assert.equal(mx.max(x).item(), 4);
    let y = mx.min(x, true);
    assert.deepEqual(y.shape, [1, 1]);
    assertArrayAllTrue(mx.equal(y, mx.array(1)));
    y = mx.max(x, true);
    assert.deepEqual(y.shape, [1, 1]);
    assertArrayAllTrue(mx.equal(y, mx.array(4)));
    assert.deepEqual(mx.min(x, 0).tolist(), [1, 2]);
    assert.deepEqual(mx.min(x, 1).tolist(), [1, 3]);
    assert.deepEqual(mx.max(x, 0).tolist(), [3, 4]);
    assert.deepEqual(mx.max(x, 1).tolist(), [2, 4]);
  });

  it('argminArgmax', () => {
    let data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let x = mx.array(data);

    assert.deepEqual(mx.argmin(x, 0, true).tolist(), [[0, 0, 0]]);
    assert.deepEqual(mx.argmin(x, 1, true).tolist(), [[0], [0], [0]]);
    assert.deepEqual(mx.argmin(x, 0).tolist(), [0, 0, 0]);
    assert.deepEqual(mx.argmin(x, 1).tolist(), [0, 0, 0]);
    assert.deepEqual(mx.argmin(x, true).tolist(), [[0]]);
    assert.equal(mx.argmin(x).item(), 0);

    assert.deepEqual(mx.argmax(x, 0, true).tolist(), [[2, 2, 2]]);
    assert.deepEqual(mx.argmax(x, 1, true).tolist(), [[2], [2], [2]]);
    assert.deepEqual(mx.argmax(x, 0).tolist(), [2, 2, 2]);
    assert.deepEqual(mx.argmax(x, 1).tolist(), [2, 2, 2]);
    assert.deepEqual(mx.argmax(x, true).tolist(), [[8]]);
    assert.equal(mx.argmax(x).item(), 8);
  });

  it('broadcast', () => {
    let a = mx.array(mx.reshape(mx.arange(200), [10, 20]));
    let b = mx.broadcastTo(a, [30, 10, 20]);
    assert.deepEqual([30, 10, 20], b.shape);
    assertArrayAllTrue(mx.equal(mx.broadcastTo(a, [30, 10, 20]), b));

    b = mx.broadcastTo(a, [1, 10, 20]);
    assert.deepEqual([1, 10, 20], b.shape);
    assertArrayAllTrue(mx.equal(mx.reshape(a, [1, 10, 20]), b));

    b = mx.broadcastTo(1, [10, 20]);
    assert.deepEqual([10, 20], b.shape);
    assertArrayAllTrue(mx.equal(mx.onesLike(b), b));
  });

  it('logsumexp', () => {
    const x = mx.array([[1.0, 2.0], [3.0, 4.0]]);
    const expected = 4.44;
    assert.closeTo(mx.logsumexp(x).item() as number, expected, 0.01);
  });

  it('mean', () => {
    const x = mx.array([[1, 2], [3, 4]]);
    assert.equal(mx.mean(x).item(), 2.5);
    const y = mx.mean(x, true);
    assert.equal(y.item(), 2.5);
    assert.deepEqual(y.shape, [1, 1]);

    assert.deepEqual(mx.mean(x, 0).tolist(), [2, 3]);
    assert.deepEqual(mx.mean(x, 1).tolist(), [1.5, 3.5]);
  });

  it('var', () => {
    let x = mx.array([[1, 2], [3, 4]]);
    assert.equal(mx.variance(x).item(), 1.25);
    let y = mx.variance(x, null, true);
    assert.equal(y.item(), 1.25);
    assert.deepEqual(y.shape, [1, 1]);

    assert.deepEqual(mx.variance(x, 0).tolist(), [1.0, 1.0]);
    assert.deepEqual(mx.variance(x, 1).tolist(), [0.25, 0.25]);

    x = mx.array([1.0, 2.0]);
    let out = mx.variance(x, null, false, 2);
    assert.equal(out.item(), Infinity);

    x = mx.array([1.0, 2.0]);
    out = mx.variance(x, null, false, 3);
    assert.equal(out.item(), Infinity);
  });

  it('std', () => {
    const x = mx.arange(10);
    const expected = 2.87;
    assert.closeTo(mx.std(x).item() as number, expected, 0.01);
  });

  it('abs', () => {
    const a = mx.array([-1.0, 1.0, -2.0, 3.0]);
    const result = mx.abs(a);
    const expected = mx.array([1.0, 1.0, 2.0, 3.0]);
    assert.deepEqual(result.tolist(), expected.tolist());
    assert.isTrue((result.tolist() as number[]).every(x => x === Math.abs(x)));
  });

  it('negative', () => {
    const a = mx.array([-1.0, 1.0, -2.0, 3.0]);
    let result = mx.negative(a);
    let expected = mx.array([1.0, -1.0, 2.0, -3.0]);
    assertArrayAllTrue(mx.equal(result, expected));
  });

  it('sign', () => {
    const a = mx.array([-1.0, 1.0, 0.0, -2.0, 3.0]);
    let result = mx.sign(a);
    let expected = mx.array([-1.0, 1.0, 0.0, -1.0, 1.0]);
    assertArrayAllTrue(mx.equal(result, expected));
  });

  it('logicalNot', () => {
    const a = mx.array([-1.0, 1.0, 0.0, 1.0, -2.0, 3.0]);
    let result = mx.logicalNot(a);
    let expected = mx.array([false, false, true, false, false, false]);
    assertArrayAllTrue(mx.equal(result, expected));
  });

  it('logicalAnd', () => {
    const a = mx.array([true, false, true, false]);
    const b = mx.array([true, true, false, false]);
    let result = mx.logicalAnd(a, b);
    let expected = mx.array([true, false, false, false]);
    assertArrayAllTrue(mx.equal(result, expected));
  });

  it('logicalOr', () => {
    const a = mx.array([true, false, true, false]);
    const b = mx.array([true, true, false, false]);
    const result = mx.logicalOr(a, b);
    const expected = mx.array([true, true, true, false]);
    assertArrayAllTrue(mx.equal(result, expected));
  });

  it('square', () => {
    const a = mx.array([0.1, 0.5, 1.0, 10.0]);
    const result = mx.square(a);
    const expected = mx.array([0.01, 0.25, 1.0, 100.0]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('sqrt', () => {
    const a = mx.array([0.1, 0.5, 1.0, 10.0]);
    const result = mx.sqrt(a);
    const expected = mx.array([0.316227766, 0.707106781, 1.0, 3.16227766]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('rsqrt', () => {
    const a = mx.array([0.1, 0.5, 1.0, 10.0]);
    const result = mx.rsqrt(a);
    const expected = mx.array([3.16227766, 1.414213562, 1.0, 0.316227766]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('reciprocal', () => {
    const a = mx.array([0.1, 0.5, 1.0, 2.0]);
    const result = mx.reciprocal(a);
    const expected = mx.array([10, 2, 1, 0.5]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('logaddexp', () => {
    let a = mx.array([0, 1, 2, 9.0]);
    let b = mx.array([1, 0, 4, 2.5]);

    let result = mx.logaddexp(a, b);
    let expected = mx.array([1.31326, 1.31326, 4.12693, 9.0015]);

    assertArrayAllTrue(mx.isclose(result, expected));

    a = mx.array([NaN]);
    b = mx.array([0.0]);
    assert.isTrue(isNaN(mx.logaddexp(a, b).item() as number));
  });

  it('log', () => {
    const a = mx.array([1, 0.5, 10, 100]);
    const result = mx.log(a);
    const expected = mx.array([0., -0.6931472, 2.3025851, 4.6051702]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('log2', () => {
    const a = mx.array([0.5, 1, 2, 10, 16]);
    const result = mx.log2(a);
    const expected = mx.array([-1., 0., 1., 3.321928, 4.]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('log10', () => {
    const a = mx.array([0.1, 1, 10, 20, 100]);
    const result = mx.log10(a);
    const expected = mx.array([-1., 0., 1., 1.30103, 2.]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('exp', () => {
    const a = mx.array([0, 0.5, -0.5, 5]);
    const result = mx.exp(a);
    const expected = mx.array([1.0, 1.6487213, 0.60653067, 148.41316]);
    assertArrayAllTrue(mx.allclose(result, expected));
  });

  it('expm1', () => {
    const a = mx.array([0, 0.5, -0.5, 5]);
    const result = mx.expm1(a);
    const expected = mx.array([0.0, 0.6487213, -0.39346933, 147.41316]);
    assertArrayAllTrue(mx.allclose(result, expected, 1e-5, 1e-5));
  });

  it('erf', () => {
    let inputs = [-5, 0.0, 0.5, 1.0, 2.0, 10.0];
    const x = mx.array(inputs);
    const expected = mx.array([-0.999999987, 0.0, 0.5205, 0.8427, 0.995322, 1.0]);
    assertArrayAllTrue(mx.isclose(mx.erf(x), expected));
  });

  it('erfinv', () => {
    let inputs = [-5.0, -1.0, 0.5, 0.0, 0.5, 1.0, 5.0];
    const x = mx.array(inputs);
    const expected = [NaN, -Infinity, 0.47693628, 0.0, 0.47693628, Infinity, NaN];
    assertArrayAllTrue(mx.allclose(mx.erfinv(x), expected, 1e-5, 1e-5, true));
  });

  it('sin', () => {
    const a = [0, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 4, 2 * Math.PI];
    const result = mx.sin(a);
    const expected = mx.array([0, 0.707107, 1, -8.74228e-08, 0.707107, 1.74846e-07]);
    assertArrayAllTrue(mx.allclose(result, expected));
  });

  it('cos', () => {
    const a = [0, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 4, 2 * Math.PI];
    const result = mx.cos(a);
    const expected = mx.array([1, 0.707107, -4.37114e-08, -1, -0.707107, 1]);
    assertArrayAllTrue(mx.allclose(result, expected));
  });

  it('log1p', () => {
    const a = mx.array([1, 0.5, 10, 100]);
    const result = mx.log1p(a);
    const expected = mx.array([0.6931472, 0.4054651, 2.3978953, 4.6151205]);
    assertArrayAllTrue(mx.allclose(result, expected));
  });

  it('sigmoid', () => {
    const a = mx.array([0.0, 1.0, -1.0, 5.0, -5.0]);
    const result = mx.sigmoid(a);
    const expected = [0.0, 1.0, -1.0, 5.0, -5.0].map(val => 1 / (1 + Math.exp(-val)));
    assertArrayAllTrue(mx.allclose(result, expected));
  });

  it('allclose', () => {
    let a = mx.array(1.0);
    let b = mx.array(1.0);
    assertArrayAllTrue(mx.allclose(a, b));

    b = mx.array(1.1);
    assertArrayAllFalse(mx.allclose(a, b));
    assertArrayAllTrue(mx.allclose(a, b, 0.1));
    assertArrayAllFalse(mx.allclose(a, b, 0.01));
    assertArrayAllTrue(mx.allclose(a, b, 0.01, 0.1));

    const c = mx.array(Infinity);
    assertArrayAllTrue(mx.allclose(c, c));
  });

  it('isclose', () => {
    let a = mx.array([Infinity, Infinity, -Infinity]);
    let b = mx.array([Infinity, -Infinity, -Infinity]);
    assert.deepEqual(mx.isclose(a, b).tolist(), [true, false, true]);

    a = mx.array([NaN]);
    assert.deepEqual(mx.isclose(a, a).tolist(), [false]);

    a = mx.array([NaN]);
    assert.deepEqual(mx.isclose(a, a, 1e-5, 1e-5, true).tolist(), [true]);
  });

  it('all', () => {
    const a = mx.array([[true, false], [true, true]]);

    assert.equal(mx.all(a).item(), false);
    assert.deepEqual(mx.all(a, true).shape, [1, 1]);
    assert.equal(mx.all(a, [0, 1]).item(), false);
    assert.deepEqual(mx.all(a, [0]).tolist(), [true, false]);
    assert.deepEqual(mx.all(a, [1]).tolist(), [false, true]);
    assert.deepEqual(mx.all(a, 0).tolist(), [true, false]);
    assert.deepEqual(mx.all(a, 1).tolist(), [false, true]);
  });

  it('any', () => {
    const a = mx.array([[true, false], [false, false]]);

    assert.equal(mx.any(a).item(), true);
    assert.deepEqual(mx.any(a, true).shape, [1, 1]);
    assert.equal(mx.any(a, [0, 1]).item(), true);
    assert.deepEqual(mx.any(a, [0]).tolist(), [true, false]);
    assert.deepEqual(mx.any(a, [1]).tolist(), [true, false]);
    assert.deepEqual(mx.any(a, 0).tolist(), [true, false]);
    assert.deepEqual(mx.any(a, 1).tolist(), [true, false]);
  });

  it('stopGradient', () => {
    const func = x => mx.sum(mx.add(mx.multiply(2, x), mx.stopGradient(mx.multiply(3, x))));

    const x = mx.array([0.0, 0.1, -3]);
    const expected = [2, 2, 2];

    assert.deepEqual(mx.grad(func)(x).tolist(), expected);
  });


  it('take', () => {
    // Shape: 4 x 3 x 2
    const l = [
      [[1, 3], [-2, -2], [-3, -2]],
      [[2, 4], [-3, 2], [-4, -2]],
      [[2, 3], [2, 4], [2, 1]],
      [[1, -5], [3, -1], [2, 3]],
    ];

    const a = mx.array(l);

    let indices = [0, -1];
    let flattenTake = mx.take(a, mx.array(indices, mx.int32)).tolist();
    assert.deepEqual(flattenTake, [1, 3]);

    indices = [-1, 2, 0];
    let axisTake = mx.take(a, mx.array(indices, mx.int32), 0).tolist();
    assert.deepEqual(axisTake, [
      [[1, -5], [3, -1], [2, 3]],
      [[2, 3], [2, 4], [2, 1]],
      [[1, 3], [-2, -2], [-3, -2]],
    ]);
  });

  it('takeAlongAxis', () => {
    let aMlx = mx.array(mx.arange(8).reshape([2, 2, 2]));
    let idxMlx = mx.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0], mx.int32);

    [null, 0, 1, 2].forEach(ax => {
      let shape;
      if (ax === null) {
        shape = [-1];
      } else {
        shape = [2, 2, 2];
        shape[ax] = 3;
      }
      let outMlx = mx.takeAlongAxis(aMlx, mx.reshape(idxMlx, shape), ax);
      let expected;
      if (ax === null) {
        expected = mx.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]).reshape([-1]);
      } else if (ax === 0) {
        expected = mx.array([[[2, 3], [0, 1]], [[4, 5], [0, 1]], [[4, 5], [4, 5]]]);
      } else if (ax === 1) {
        expected = mx.array([[[2, 1], [0, 3], [2, 1]], [[4, 5], [4, 7], [6, 5]]]);
      } else if (ax === 2) {
        expected = mx.array([[[1, 0, 0], [3, 3, 2]], [[4, 4, 4], [7, 7, 6]]]);
      }
      assertArrayAllTrue(mx.equal(expected, outMlx));
    });
  });

  it('split', () => {
    let a = mx.array([1, 2, 3]);
    let splits = mx.split(a, 3);
    splits.forEach((x, e) => {
      assert.equal(x.item(), e + 1);
    });

    a = mx.array([[1, 2], [3, 4], [5, 6]]);
    const [x, y, z] = mx.split(a, 3, 0);
    assert.deepEqual(x.tolist(), [[1, 2]]);
    assert.deepEqual(y.tolist(), [[3, 4]]);
    assert.deepEqual(z.tolist(), [[5, 6]]);

    assert.throws(() => {
      mx.split(a, 3, 2);
    }, Error);

    a = mx.arange(8);
    const [x1, y1, z1] = mx.split(a, [1, 5]);
    assert.deepEqual(x1.tolist(), [0]);
    assert.deepEqual(y1.tolist(), [1, 2, 3, 4]);
    assert.deepEqual(z1.tolist(), [5, 6, 7]);
  });

  it('arangeOverloadDispatch', () => {
    assert.throws(() => mx.arange(Number.NaN, 1, 5));
    assert.throws(() => mx.arange(0, Number.NaN, 5));
    assert.throws(() => mx.arange(0, 2, Number.NaN));
    assert.throws(() => mx.arange(0, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY));
    assert.throws(() => mx.arange(Number.POSITIVE_INFINITY, 1, Number.POSITIVE_INFINITY));
    assert.throws(() => mx.arange(Number.POSITIVE_INFINITY, 1, 5));
    assert.throws(() => {
      const intMax = 2147483647;
      mx.arange(0, intMax + 1, 1);
    });

    let a = mx.arange(5);
    let expected = [0, 1, 2, 3, 4];
    assert.deepEqual(a.tolist(), expected);

    a = mx.arange(1, 5);
    expected = [1, 2, 3, 4];
    assert.deepEqual(a.tolist(), expected);

    a = mx.arange(3);
    expected = [0, 1, 2];
    assert.deepEqual(a.tolist(), expected);
  });

  it('arangeInferredDtype', () => {
    let a = mx.arange(5);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(5.0);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1, 3.0);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1, 3, 1, mx.float32);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1, 5, 1);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1.0, 5, 1);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1, 5.0, 1);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1, 5, 1.0);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(1.0, 3.0, 0.2, mx.int32);
    assert.equal(a.dtype, mx.int32);
  });

  it('arangeCornerCasesCast', () => {
    let a = mx.arange(0, 3, 0.2, mx.int32);
    let expected = Array(15).fill(0);
    assert.deepEqual(a.tolist(), expected);
    assert.equal(a.dtype, mx.int32);

    a = mx.arange(-1, -4, -0.9, mx.int32);
    expected = Array(4).fill(-1);
    assert.deepEqual(a.tolist(), expected);
    assert.equal(a.dtype, mx.int32);

    a = mx.arange(-1, -20, -1.2, mx.int32);
    expected = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16];
    assert.deepEqual(a.tolist(), expected);
    assert.equal(a.dtype, mx.int32);

    a = mx.arange(0, 10, 100);
    expected = [0];
    assert.deepEqual(a.tolist(), expected);
    assert.equal(a.dtype, mx.float32);

    a = mx.arange(10, 0, 1);
    expected = [];
    assert.deepEqual(a.tolist(), expected);

    a = mx.arange(10, 0, Number.POSITIVE_INFINITY);
    expected = [];
    assert.deepEqual(a.tolist(), expected);

    a = mx.arange(0, 10, Number.POSITIVE_INFINITY);
    expected = [0];
    assert.deepEqual(a.tolist(), expected);

    a = mx.arange(0, -10, Number.NEGATIVE_INFINITY);
    expected = [0];
    assert.deepEqual(a.tolist(), expected);
  });

  describe('unaryOps', () => {
    const testOps = (tfOp, mlxOp, x, y, atol) => {
      const rTf = tfOp(x);
      const rMlx = mlxOp(y);
      mx.eval(rMlx);

      assertArrayAllTrue(mx.isclose(rMlx, rTf.arraySync(), atol));
    };

    const x = tf.randomNormal([18, 28, 38]);
    const ops = ['abs', 'exp', 'log', 'square', 'sqrt'];

    for (let op of ops) {
      it(op, () => {
        const x_ = tf.cast(x, 'float32');
        const y_ = mx.array(x_.arraySync());
        testOps(tf[op], mx[op], x_, y_, 1e-6);
      });
    }
  });

  describe('trigOps', () => {
    const testOps = (tfOp, mlxOp, x, y, atol) => {
      const rTf = tfOp(x);
      const rMlx = mlxOp(y);
      mx.eval(rMlx);

      assertArrayAllTrue(mx.isclose(rMlx, rTf.arraySync(), atol));
    };

    const x = tf.randomNormal([9, 12, 18]).arraySync();
    const xi = tf.randomNormal([9, 12, 18]).arraySync();
    const baseOps = ['sin', 'cos', 'tan'];
    const hyperbolicOps = ['sinh', 'cosh', 'tanh'];
    const allFwdOps = baseOps.concat(hyperbolicOps);

    for (let op of allFwdOps) {
      it(op, () => {
        const tArr = tf.tensor(x);
        const mArr = mx.array(x);
        testOps(tf[op], mx[op], tArr, mArr, 1e-6);
      });
    }

    describe('grad', () => {
      for (let op of allFwdOps) {
        it(op, () => {
          const primalT = tf.tensor(xi);
          const primalM = mx.array(primalT.arraySync());
          const tArr = tf.tensor(x);
          const mArr = mx.array(tArr.arraySync());

          const tfVjp = tf.grad(tf[op]);
          const mlxVjp = (cotan) => mx.vjp(mx[op], [primalM], [cotan])[1][0];
          testOps(tfVjp, mlxVjp, tArr, mArr, 1e-5);

          const tfOpFwd = tf[op];
          let primalTInv = tfOpFwd(tf.tensor(xi));

          if (op == 'cosh') {
            primalTInv = tf.add(primalTInv, tf.fill(primalTInv.shape, 1e-3));
          } else if (op == 'cos') {
            primalTInv = tf.sub(primalTInv, tf.fill(primalTInv.shape, 1e-3));
          }

          const primalMInv = mx.array(primalTInv.arraySync());
          const tfVjpInv = tf.grad(tf[op]);
          const mlxVjpInv = (cotan) => mx.vjp(mx[op], [primalMInv], [cotan])[1][0];
          testOps(tfVjpInv, mlxVjpInv, tf.tensor(x).arraySync(), mx.array(x), 1e-5);
        });
      }
    });
  });

  describe('binaryOps', () => {
    const testOps = (tfOp, mlxOp, x1, x2, y1, y2, atol) => {
      let rTf = tfOp(x1, x2);
      let rMlx = mlxOp(y1, y2);
      mx.eval(rMlx);
      assertArrayAllTrue(mx.allclose(rMlx, rTf.arraySync(), atol));
    };

    const x1 = tf.maximum(tf.randomNormal([18, 28, 38]), 0.1);
    const x2 = tf.maximum(tf.randomNormal([18, 28, 38]), 0.1);
    const y1 = mx.array(x1.arraySync());
    const y2 = mx.array(x2.arraySync());

    const ops = {
      'add': 'add',
      'subtract': 'sub',
      'multiply': 'mul',
      'divide': 'div',
      'floorDivide': 'floorDiv',
      'maximum': 'maximum',
      'minimum': 'minimum',
      'power': 'pow'
    };

    for (let [npOp, tfOp] of Object.entries(ops)) {
      it(npOp, () => {
        const x1_ = x1.toFloat();
        const x2_ = x2.toFloat();
        const y1_ = mx.array(x1_.arraySync());
        const y2_ = mx.array(x2_.arraySync());

        testOps(tf[tfOp], mx[npOp], x1_, x2_, y1_, y2_, 1e-6);
      });
    }
  });

  describe('irregularBinaryOps', () => {
    const dims = [2, 3, 4, 5];
    const size = 3;
    const trialMul = 2;

    for (let d of dims) {
      it(`${d}Dim`, () => {
        let anp = tf.randomUniform([Math.pow(size, d)], -20, 20, 'int32')
                                .reshape(Array(d).fill(size));
        let bnp = tf.randomUniform([Math.pow(size, d)], -20, 20, 'int32')
                                .reshape(Array(d).fill(size));
        for (let i = 0; i < trialMul * d; i++) {
          const amlx = mx.array(anp.arraySync());
          const bmlx = mx.array(bnp.arraySync());
          const aT = Array.from(tf.util.createShuffledIndices(d));
          const bT = Array.from(tf.util.createShuffledIndices(d));
          let outnp = tf.add(anp.transpose(aT), bnp.transpose(bT));
          let outmlx = mx.add(mx.transpose(amlx, aT), mx.transpose(bmlx, bT));

          assert.deepEqual(outnp.arraySync(), outmlx.tolist());
        }
      });
    }

    for (let d of dims) {
      it(`${d}DimBroadcast`, () => {
        let anp = tf.randomUniform([Math.pow(size, d)], -20, 20, 'int32')
                                .reshape(Array(d).fill(size));
        for (let nBsx = 0; nBsx < d; nBsx++) {
          let bnp = tf.randomUniform([Math.pow(size, nBsx)], -20, 20, 'int32')
                                  .reshape(Array(nBsx).fill(size));
          for (let i = 0; i < trialMul * d; i++) {
            const amlx = mx.array(anp.arraySync());
            const bmlx = mx.array(bnp.arraySync());
            const bShape = Array(d - nBsx).fill(1).concat(Array(nBsx).fill(size));
            tf.util.shuffle(bShape);

            let outnp = tf.add(anp, bnp.reshape(bShape));
            let outmlx = mx.add(amlx, mx.reshape(bmlx, bShape));
            assert.deepEqual(outnp.arraySync(), outmlx.tolist());
          }
        }
      });
    }
  });

  it('concatenate', function() {
    this.timeout(10 * 1000);  // slow in QEMU

    const aTf = tf.randomNormal([32, 32, 32]);
    const bTf = tf.randomNormal([32, 32, 32]);
    const aMlx = mx.array(aTf.arraySync());
    const bMlx = mx.array(bTf.arraySync());

    const axes = [0, 1, 2];
    const permutations = [
      [0, 1, 2],
      [0, 2, 1],
      [1, 0, 2],
      [1, 2, 0],
      [2, 0, 1],
      [2, 1, 0],
    ];

    for (let axis of axes) {
      for (let p of permutations) {
        const cTf = tf.concat([aTf, tf.transpose(bTf, p)], axis);
        const cMlx = mx.concatenate([aMlx, mx.transpose(bMlx, p)], axis);
        assert.equal(cTf.shape.toString(), cMlx.shape.toString());
        assertArrayAllTrue(mx.isclose(cMlx, cTf.arraySync()));
      }
    }

    assert.throws(() => {
      const a = mx.array([[1, 2], [1, 2], [1, 2]]);
      const b = mx.array([1, 2]);
      mx.concatenate([a, b], 0);
    }, Error);
  });

  describe('meshgrid', () => {
    it('singleInput', () => {
      const x = mx.array([1, 2, 3], mx.int32);
      const y = tf.tensor([1, 2, 3], null, 'int32');

      const aMlx = mx.meshgrid(x);
      const aNp = tf.meshgrid(y);
      assert.deepEqual(aMlx[0].tolist(), aNp[0].arraySync());
    });

    it('differentLengths', () => {
      let x = mx.array([1, 2], mx.int32);
      let y = mx.array([1, 2, 3], mx.int32);
      let z = tf.tensor([1, 2], null, 'int32');
      let w = tf.tensor([1, 2, 3], null, 'int32');
      let [aMlx, bMlx] = mx.meshgrid(x, y);
      let [aNp, bNp] = tf.meshgrid(z, w);
      assert.deepEqual(aMlx.tolist(), aNp.arraySync());
      assert.deepEqual(bMlx.tolist(), bNp.arraySync());
    });

    it('emptyInput', () => {
      let x = mx.array([], mx.int32);
      let y = tf.tensor([], null, 'int32');
      let aMlx = mx.meshgrid(x);
      let aNp = tf.meshgrid(y);
      assert.deepEqual(aMlx[0].tolist(), aNp[0].arraySync());
    });

    it('float32Input', () => {
      let x = mx.array([1.1, 2.2, 3.3], mx.float32);
      let y = tf.tensor([1.1, 2.2, 3.3], null, 'float32');
      let aMlx = mx.meshgrid(x, x);
      let aNp = tf.meshgrid(y, y);
      assert.deepEqual(aMlx[0].tolist(), aNp[0].arraySync());
      assert.deepEqual(aMlx[1].tolist(), aNp[1].arraySync());
    });
  });

  it('pad', () => {
    const padWidthAndValues = [
      [[1, 1], [1, 1]], 0,
      [[1, 1], [1, 1]], 5,
      [[3, 0], [0, 2]], 0,
      [[3, 0], [0, 2]], -7,
      [[0, 0], [0, 0]], 0,
    ];

    for (let i = 0; i < padWidthAndValues.length; i += 2) {
      const pw = padWidthAndValues[i] as [number, number][];
      const v = padWidthAndValues[i + 1] as number;

      const aNpy = tf.randomNormal([16, 16]);
      const aMlx = mx.array(aNpy.arraySync());

      const bNpy = tf.pad(aNpy, pw, v);
      const bMlx = mx.pad(aMlx, pw, v);

      assert.deepEqual(bNpy.shape, bMlx.shape);
      assertArrayAllTrue(mx.isclose(bNpy.arraySync(), bMlx));
    }

    const a = mx.zeros([1, 1, 1]);
    assert.deepEqual(mx.pad(a, 1).shape, [3, 3, 3]);
    assert.deepEqual(mx.pad(a, [1]).shape, [3, 3, 3]);
    assert.deepEqual(mx.pad(a, [1, 2]).shape, [4, 4, 4]);
    assert.deepEqual(mx.pad(a, [[1, 2]]).shape, [4, 4, 4]);
    assert.deepEqual(mx.pad(a, [[1, 2], [2, 1], [2, 2]]).shape, [4, 4, 5]);

    const aFwd = mx.array(tf.randomUniform([16, 16]).arraySync());
    const aBwd = mx.ones([22, 22]);
    const f = x => mx.pad(x, [[4, 2], [2, 4]]);

    const [_, df] = mx.vjp(f, [aFwd], [aBwd]);
    assert.deepEqual(df[0].shape, [16, 16]);
  });

  it('where', () => {
    const rMlx1 = mx.where(true, mx.array([[1, 2], [3, 4]]), 1);
    const rTf1 = tf.where(tf.tensor(true), tf.tensor([[1, 2], [3, 4]]), tf.tensor(1));
    assert.deepEqual(rMlx1.tolist(), rTf1.arraySync());

    const rMlx2 = mx.where(true, 1, mx.array([[1, 2], [3, 4]]));
    const rTf2 = tf.where(tf.tensor(true), tf.tensor(1), tf.tensor([[1, 2], [3, 4]]));
    assert.deepEqual(rMlx2.tolist(), rTf2.arraySync());

    const rMlx3 = mx.where(
      mx.array([[true, false], [false, true]]),
      mx.array([[1, 2], [3, 4]]),
      mx.array([5, 6])
    );
    const rTf3 = tf.where(
      tf.tensor([[true, false], [false, true]]),
      tf.tensor([[1, 2], [3, 4]]),
      tf.tensor([5, 6])
    );
    assert.deepEqual(rMlx3.tolist(), rTf3.arraySync());
  });

  it('asStrided', () => {
    const x = mx.random.normal([128]).astype(mx.float32);
    const shapes = [[10, 10], [5, 5], [2, 20], [10]];
    const strides = [[3, 3], [7, 1], [1, 5], [4]];
    for (let i = 0; i < shapes.length; i++) {
      const shape = shapes[i];
      const stride = strides[i];
      for (let offset of [0, 1, 3]) {
        const y = mx.asStrided(x, shape, stride, offset);
        assert.deepEqual(y.shape, shape);
      }
    }
  });

  it('squeezeExpand', () => {
    let a = mx.zeros([2, 1, 2, 1]);
    assert.deepEqual(mx.squeeze(a).shape, [2, 2]);
    assert.deepEqual(mx.squeeze(a, 1).shape, [2, 2, 1]);
    assert.deepEqual(mx.squeeze(a, [1, 3]).shape, [2, 2]);
    assert.deepEqual(a.squeeze().shape, [2, 2]);
    assert.deepEqual(a.squeeze(1).shape, [2, 2, 1]);
    assert.deepEqual(a.squeeze([1, 3]).shape, [2, 2]);

    a = mx.zeros([2, 2]);
    assert.deepEqual(mx.squeeze(a).shape, [2, 2]);

    assert.deepEqual(mx.expandDims(a, 0).shape, [1, 2, 2]);
    assert.deepEqual(mx.expandDims(a, [0, 1]).shape, [1, 1, 2, 2]);
    assert.deepEqual(mx.expandDims(a, [0, -1]).shape, [1, 2, 2, 1]);
  });

  it('sort', () => {
    const x = mx.array([3, 1, 2]);
    const sortedX = mx.sort(x);
    assert.deepEqual(sortedX.tolist(), [1, 2, 3]);
  });

  it('largeBinary', () => {
    const a = mx.ones([1000, 214748], mx.int8);
    const b = mx.ones([214748], mx.int8);
    assert.equal(mx.add(a, b).index(0, 0).item(), 2);
  });

  it('eye', () => {
    const size = 5;
    const eyeMatrix = mx.eye(size);

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (i === j) {
          assert.equal(eyeMatrix.index(i, j).item(), 1);
        } else {
          assert.equal(eyeMatrix.index(i, j).item(), 0);
        }
      }
    }
  });

  it('stack', () => {
    let a = mx.ones([2]);
    let tfA = tf.tensor([1, 1]);
    let b = mx.ones([2]);
    let tfB = tf.tensor([1, 1]);

    let c = mx.stack([a, b]);
    let tfC = tf.stack([tfA, tfB]);
    assert.deepEqual(c.tolist(), tfC.arraySync());

    c = mx.stack([a, b], 1);
    tfC = tf.stack([tfA, tfB], 1);
    assert.deepEqual(c.tolist(), tfC.arraySync());

    a = mx.ones([1, 2]);
    tfA = tf.tensor([[1, 1]]);
    b = mx.ones([1, 2]);
    tfB = tf.tensor([[1, 1]]);

    c = mx.stack([a, b]);
    tfC = tf.stack([tfA, tfB]);
    assert.deepEqual(c.tolist(), tfC.arraySync());

    c = mx.stack([a, b], 1);
    tfC = tf.stack([tfA, tfB], 1);
    assert.deepEqual(c.tolist(), tfC.arraySync());
  });

  it('flatten', () => {
    const x = mx.zeros([2, 3, 4]);
    assert.deepEqual(mx.flatten(x).shape, [2 * 3 * 4]);
    assert.deepEqual(mx.flatten(x, 1).shape, [2, 3 * 4]);
    assert.deepEqual(mx.flatten(x, null, 1).shape, [2 * 3, 4]);
    assert.deepEqual(x.flatten().shape, [2 * 3 * 4]);
    assert.deepEqual(x.flatten(1).shape, [2, 3 * 4]);
    assert.deepEqual(x.flatten(null, 1).shape, [2 * 3, 4]);
  });

  it('clip', () => {
    let a = tf.tensor([1, 4, 3, 8, 5]);
    let expected = a.clipByValue(2, 6);
    let clipped = mx.clip(mx.array(a.arraySync()), 2, 6);
    assert.deepEqual(clipped.tolist(), expected.arraySync());

    a = tf.tensor([-1, 1, 0, 5]);
    expected = a.clipByValue(0, Infinity);
    clipped = mx.clip(mx.array(a.arraySync()), 0, Infinity);
    assert.deepEqual(clipped.tolist(), expected.arraySync());

    a = tf.tensor([2, 3, 4, 5]);
    expected = a.clipByValue(-Infinity, 4);
    clipped = mx.clip(mx.array(a.arraySync()), -Infinity, 4);
    assert.deepEqual(clipped.tolist(), expected.arraySync());

    clipped = mx.clip(mx.array(a.arraySync()), [3, 1, 5, 5], 4);
    assert.deepEqual(clipped.tolist(), [3, 3, 4, 4]);

    clipped = mx.clip(mx.array(a.arraySync()), [3, 1, 5, 5], [5, -1, 2, 9]);
    assert.deepEqual(clipped.tolist(), [3, -1, 2, 5]);
  });

  it('linspace', () => {
    let a = mx.linspace(0, 1);
    let expected = tf.linspace(0, 1, 50).arraySync();
    assertArrayAllTrue(mx.isclose(a, expected));

    let b = mx.linspace(0, 10, 5, mx.int64);
    expected = tf.linspace(0, 10, 5).toInt().arraySync();
    assertArrayAllTrue(mx.isclose(b, expected));

    let c = mx.linspace(-2.7, -0.7, 7);
    expected = tf.linspace(-2.7, -0.7, 7).arraySync();
    assertArrayAllTrue(mx.isclose(c, expected));

    let d = mx.linspace(0, 1, 10);
    expected = tf.linspace(0, 1, 10).arraySync();
    assertArrayAllTrue(mx.isclose(d, expected));

    d = mx.linspace(1, 10, 1);
    expected = tf.linspace(1, 10, 1).arraySync();
    assertArrayAllTrue(mx.isclose(d, expected));
  });

  it('repeat', () => {
    let array = mx.array([1, 2, 3]);

    let repeatedArray = mx.repeat(array, 2);
    assert.deepEqual(repeatedArray.tolist(), [1, 1, 2, 2, 3, 3]);

    repeatedArray = mx.repeat(array, 3, 0);
    assert.deepEqual(repeatedArray.tolist(), [1, 1, 1, 2, 2, 2, 3, 3, 3]);

    array = mx.array([[1, 2], [3, 4]]);
    repeatedArray = mx.repeat(array, 2, 1);
    assert.deepEqual(repeatedArray.tolist(), [[1, 1, 2, 2], [3, 3, 4, 4]]);

    repeatedArray = mx.repeat(array, 3, 0);
    assert.deepEqual(repeatedArray.tolist(), [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]);
  });

  it('emptyMatmuls', () => {
    let a = mx.array([]);
    let b = mx.array([]);
    assert.equal(mx.inner(a, b).item(), 0.0);

    a = mx.zeros([10, 0]);
    b = mx.zeros([0, 10]);
    const out = mx.matmul(a, b);
    assertArrayAllTrue(mx.arrayEqual(out, mx.zeros([10, 10])));
  });

  it('diagonal', () => {
    const x = mx.array([
      [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
      [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
    ]);
    const expected = [[0, 13], [4, 17], [8, 21]];
    assert.deepEqual(mx.diagonal(x, 0, -1, 0).tolist(), expected);

    const expected2 = [[1, 14], [5, 18], [9, 22]];
    assert.deepEqual(mx.diagonal(x, -1, 2, 0).tolist(), expected2);
  });

  it('diag', () => {
    let x = mx.array([1, 2, 3, 4]);
    let expected = mx.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]);
    let result = mx.diag(x);
    assertArrayAllTrue(mx.arrayEqual(result, expected));

    x = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expected = mx.array([1, 5, 9]);
    result = mx.diag(x);
    assertArrayAllTrue(mx.arrayEqual(result, expected));

    expected = mx.array([2, 6]);
    result = mx.diag(x, 1);
    assertArrayAllTrue(mx.arrayEqual(result, expected));
  });
});
