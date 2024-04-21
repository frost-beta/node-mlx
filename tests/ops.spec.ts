import mx from '..';
import {assertArrayAllTrue, assertArrayNotAllTrue, assertArrayAllFalse} from './utils';
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
});
