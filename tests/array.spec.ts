import {core as mx} from '..';
import {assertArray, assertArrayAllTrue, assertArrayAllFalse} from './utils';
import {assert} from 'chai';

describe('dtype', () => {
  it('size', () => {
    assert.equal(mx.bool_.size, 1);
    assert.equal(mx.uint8.size, 1);
    assert.equal(mx.uint16.size, 2);
    assert.equal(mx.uint32.size, 4);
    assert.equal(mx.uint64.size, 8);
    assert.equal(mx.int8.size, 1);
    assert.equal(mx.int16.size, 2);
    assert.equal(mx.int32.size, 4);
    assert.equal(mx.int64.size, 8);
    assert.equal(mx.float16.size, 2);
    assert.equal(mx.float32.size, 4);
    assert.equal(mx.bfloat16.size, 2);
    assert.equal(mx.complex64.size, 8);
  });
});

describe('equality', () => {
  it('arrayEqArray', () => {
    const a = mx.array([1, 2, 3]);
    const b = mx.array([1, 2, 3]);
    const c = mx.array([1, 2, 4]);
    assertArrayAllTrue(mx.all(mx.equal(a, b)));
    assertArrayAllFalse(mx.all(mx.equal(a, c)));
  });

  it('arrayEqScalar', () => {
    const a = mx.array([1, 2, 3]);
    const b = 1;
    const c = 4;
    const d = 2.5;
    const e = mx.array([1, 2.5, 3.25]);
    assertArrayAllTrue(mx.any(mx.equal(a, b)));
    assertArrayAllFalse(mx.all(mx.equal(a, c)));
    assertArrayAllFalse(mx.all(mx.equal(a, d)));
    assertArrayAllTrue(mx.any(mx.equal(a, e)));
  });

  it('listEqualsArray', () => {
    const a = mx.array([1, 2, 3]);
    const b = [1, 2, 3];
    const c = [1, 2, 4];
    assertArrayAllTrue(mx.equal(a, b));
    assertArray(mx.equal(a, c),
                (arrays) => assert.deepEqual(arrays, [true, true, false]));
  });
});

describe('inequality', () => {
  it('arrayNeArray', () => {
    const a = mx.array([1, 2, 3]);
    const b = mx.array([1, 2, 3]);
    const c = mx.array([1, 2, 4]);
    assertArrayAllFalse(mx.any(mx.notEqual(a, b)));
    assertArrayAllTrue(mx.any(mx.notEqual(a, c)));
  });

  it('arrayNeScalar', () => {
    const a = mx.array([1, 2, 3]);
    const b = 1;
    const c = 4;
    const d = 1.5;
    const e = 2.5;
    const f = mx.array([1, 2.5, 3.25]);
    assertArrayAllFalse(mx.all(mx.notEqual(a, b)));
    assertArrayAllTrue(mx.any(mx.notEqual(a, c)));
    assertArrayAllTrue(mx.any(mx.notEqual(a, d)));
    assertArrayAllTrue(mx.any(mx.notEqual(a, e)));
    assertArrayAllFalse(mx.all(mx.notEqual(a, f)));
  });

  it('listNotEqualsArray', () => {
    const a = mx.array([1, 2, 3]);
    const b = [1, 2, 3];
    const c = [1, 2, 4];
    assertArrayAllFalse(mx.notEqual(a, b));
    assertArray(mx.notEqual(a, c),
                (arrays) => assert.deepEqual(arrays, [false, false, true]));
  });
});

describe('array', () => {
  it('arrayBasics', () => {
    let x = mx.array(1);
    assert.equal(x.size, 1);
    assert.equal(x.ndim, 0);
    assert.equal(x.itemsize, 4);
    assert.equal(x.nbytes, 4);
    assert.deepEqual(x.shape, []);
    assert.equal(x.dtype, mx.float32);
    assert.equal(x.item(), 1);
    assert.isTrue(typeof x.item() === 'number');

    assert.throws(() => {
      x.length;
    }, TypeError);

    x = mx.array(1, mx.uint32);
    assert.equal(x.item(), 1);
    assert.isTrue(typeof x.item() === 'number');

    x = mx.array(1, mx.int64);
    assert.equal(x.item(), 1);
    assert.isTrue(typeof x.item() === 'number');

    x = mx.array(1, mx.bfloat16);
    assert.equal(x.item(), 1.0);

    x = mx.array(1.0);
    assert.equal(x.size, 1);
    assert.equal(x.ndim, 0);
    assert.deepEqual(x.shape, []);
    assert.equal(x.dtype, mx.float32);
    assert.equal(x.item(), 1.0);
    assert.isTrue(typeof x.item() === 'number');

    x = mx.array(false);
    assert.equal(x.size, 1);
    assert.equal(x.ndim, 0);
    assert.deepEqual(x.shape, []);
    assert.equal(x.dtype, mx.bool_);
    assert.equal(x.item(), false);
    assert.isTrue(typeof x.item() === 'boolean');

    x = mx.array(mx.Complex(1, 1));
    assert.equal(x.ndim, 0);
    assert.deepEqual(x.shape, []);
    assert.equal(x.dtype, mx.complex64);
    assert.deepEqual(x.item(), {re: 1, im: 1});
    assert.isTrue(typeof x.item() === 'object');

    x = mx.array([true, false, true]);
    assert.equal(x.dtype, mx.bool_);
    assert.equal(x.ndim, 1);
    assert.deepEqual(x.shape, [3]);
    assert.equal(x.length, 3);

    x = mx.array([true, false, true], mx.float32);
    assert.equal(x.dtype, mx.float32);

    x = mx.array([0, 1, 2]);
    assert.equal(x.dtype, mx.float32);
    assert.equal(x.ndim, 1);
    assert.deepEqual(x.shape, [3]);

    x = mx.array([0, 1, 2], mx.float32);
    assert.equal(x.dtype, mx.float32);

    x = mx.array([0.0, 1.0, 2.0]);
    assert.equal(x.dtype, mx.float32);
    assert.equal(x.ndim, 1);
    assert.deepEqual(x.shape, [3]);

    x = mx.array([mx.Complex(0, 1), mx.Complex(1)]);
    assert.equal(x.dtype, mx.complex64);
    assert.equal(x.ndim, 1);
    assert.deepEqual(x.shape, [2]);

    x = mx.array([1, 2, 3], mx.int32);
    assert.equal(x.dtype, mx.int32);
    assert.deepEqual(x.tolist(), [1, 2, 3]);
  });

  it('boolConversion', () => {
    let x = mx.array(true);
    assertArrayAllTrue(x);
    x = mx.array(false);
    assertArrayAllFalse(x);
  });

  it('constructionFromLists', () => {
    let x = mx.array([]);
    assert.equal(x.size, 0);
    assert.deepEqual(x.shape, [0]);
    assert.equal(x.dtype, mx.float32);

    x = mx.array([[], [], []]);
    assert.equal(x.size, 0);
    assert.deepEqual(x.shape, [3, 0]);
    assert.equal(x.dtype, mx.float32);

    x = mx.array([[[], []], [[], []], [[], []]]);
    assert.equal(x.size, 0);
    assert.deepEqual(x.shape, [3, 2, 0]);
    assert.equal(x.dtype, mx.float32);

    assert.throws(() => {
      x = mx.array([[[], []], [[]], [[], []]]);
    }, Error);

    assert.throws(() => {
      x = mx.array([[[], []], [[1.0, 2.0], []], [[], []]]);
    }, Error);

    assert.throws(() => {
      x = mx.array([[0, 1], [[0, 1], 1]]);
    }, Error);

    x = mx.array([[1.0, 2.0], [0.0, 3.9]], mx.bool_);
    assert.equal(x.dtype, mx.bool_);
    assertArrayAllTrue(mx.arrayEqual(x, mx.array([[true, true], [false, true]])));

    x = mx.array([[1.0, 2.0], [0.0, 3.9]], mx.int32);
    assertArrayAllTrue(mx.arrayEqual(x, mx.array([[1, 2], [0, 3]])));

    x = mx.array([mx.Complex(1, 0), mx.Complex(0, 2)], mx.complex64);
    assert.deepEqual(x.tolist(), [{re: 1, im: 0}, {re: 0, im: 2}]);
  });

  it('arrayToList', () => {
    const types = [mx.bool_, mx.uint32, mx.int32, mx.int64, mx.float32];
    for (const t of types) {
      const xSingle = mx.array(1, t);
      assertArrayAllTrue(mx.equal(xSingle.tolist(), 1));
    }

    const valsMultiple = [1, 2, 3, 4];
    const xMultiple = mx.array(valsMultiple);
    assertArrayAllTrue(mx.equal(xMultiple.tolist(), valsMultiple));

    const vals2D = [[1, 2], [3, 4]];
    const x2D = mx.array(vals2D);
    assertArrayAllTrue(mx.equal(x2D.tolist(), vals2D));

    const valsBool = [[1, 0], [0, 1]];
    const xBool = mx.array(valsBool, mx.bool_);
    assertArrayAllTrue(mx.equal(xBool.tolist(), valsBool));

    const valsFloat = [[1.5, 2.5], [3.5, 4.5]];
    const xFloat = mx.array(valsFloat);
    assertArrayAllTrue(mx.equal(xFloat.tolist(), valsFloat));

    const vals3D = [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]];
    const x3D = mx.array(vals3D);
    assertArrayAllTrue(mx.equal(x3D.tolist(), vals3D));

    const valsEmpty = [];
    const xEmpty = mx.array(valsEmpty);
    assertArrayAllTrue(mx.equal(xEmpty.tolist(), valsEmpty));

    const valsEmpty2D = [[], []];
    const xEmpty2D = mx.array(valsEmpty2D);
    assertArrayAllTrue(mx.equal(xEmpty2D.tolist(), valsEmpty2D));

    const valsHalf = [1.0, 2.0, 3.0, 4.0, 5.0];
    const xHalfFloat16 = mx.array(valsHalf, mx.float16);
    assertArrayAllTrue(mx.equal(xHalfFloat16.tolist(), valsHalf));

    const xHalfBfloat16 = mx.array(valsHalf, mx.bfloat16);
    assertArrayAllTrue(mx.equal(xHalfBfloat16.tolist(), valsHalf));
  });

  it('dtypeJSScalarPromotion', () => {
    const tests: [mx.Dtype, (a, b) => mx.array, any, mx.Dtype][] = [
      [mx.bool, mx.multiply, false, mx.bool],
      [mx.bool, mx.multiply, 0, mx.float32],
      [mx.bool, mx.multiply, 1.0, mx.float32],
      [mx.int8, mx.multiply, false, mx.int8],
      [mx.int8, mx.multiply, 0, mx.float32],
      [mx.int8, mx.multiply, 1.0, mx.float32],
      [mx.int16, mx.multiply, false, mx.int16],
      [mx.int16, mx.multiply, 0, mx.float32],
      [mx.int16, mx.multiply, 1.0, mx.float32],
      [mx.int32, mx.multiply, false, mx.int32],
      [mx.int32, mx.multiply, 0, mx.float32],
      [mx.int32, mx.multiply, 1.0, mx.float32],
      [mx.int64, mx.multiply, false, mx.int64],
      [mx.int64, mx.multiply, 0, mx.float32],
      [mx.int64, mx.multiply, 1.0, mx.float32],
      [mx.uint8, mx.multiply, false, mx.uint8],
      [mx.uint8, mx.multiply, 0, mx.float32],
      [mx.uint8, mx.multiply, 1.0, mx.float32],
      [mx.uint16, mx.multiply, false, mx.uint16],
      [mx.uint16, mx.multiply, 0, mx.float32],
      [mx.uint16, mx.multiply, 1.0, mx.float32],
      [mx.uint32, mx.multiply, false, mx.uint32],
      [mx.uint32, mx.multiply, 0, mx.float32],
      [mx.uint32, mx.multiply, 1.0, mx.float32],
      [mx.uint64, mx.multiply, false, mx.uint64],
      [mx.uint64, mx.multiply, 0, mx.float32],
      [mx.uint64, mx.multiply, 1.0, mx.float32],
      [mx.float32, mx.multiply, false, mx.float32],
      [mx.float32, mx.multiply, 0, mx.float32],
      [mx.float32, mx.multiply, 1.0, mx.float32],
      [mx.float16, mx.multiply, false, mx.float16],
      [mx.float16, mx.multiply, 0, mx.float16],
      [mx.float16, mx.multiply, 1.0, mx.float16],
    ];

    for (const [dtypeIn, f, v, dtypeOut] of tests) {
      const x = mx.array(0, dtypeIn);
      const y = f(x, v);
      assert.equal(y.dtype, dtypeOut);
    }
  });

  it('arrayTypeCast', () => {
    const a = mx.array([0.1, 2.3, -1.3]);
    const b = [0, 2, -1];

    assert.deepEqual(a.astype(mx.int32).tolist(), b);
    assert.equal(a.astype(mx.int32).dtype, mx.int32);

    const c = mx.array(b).astype(mx.float32);
    assert.equal(c.dtype, mx.float32);
  });

  it('arrayIteration', () => {
    let a = mx.array([0, 1, 2]);
    let i = 0;
    for (const el of a) {
      assert.equal(el.item(), i++);
    }

    a = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let [x, y, z] = a;
    assert.deepEqual(x.tolist(), [1.0, 2.0]);
    assert.deepEqual(y.tolist(), [3.0, 4.0]);
    assert.deepEqual(z.tolist(), [5.0, 6.0]);
  });

  it('arrayCopy', () => {
    const dtypes = [
      mx.int8,
      mx.int16,
      mx.int32,
      mx.int64,
      mx.uint8,
      mx.uint16,
      mx.uint32,
      mx.uint64,
      mx.float16,
      mx.float32,
      mx.bfloat16,
      mx.complex64,
    ];

    for (const dtype of dtypes) {
      const x = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype);
      let y = new mx.array(x);
      assertArrayAllTrue(mx.equal(y, x));

      y = mx.add(y, -1);
      assertArrayAllTrue(mx.equal(y, mx.add(x, -1)));
    }
  });

  describe('indexing', () => {
    it('ellipsis', () => {
      const a = mx.array([1]).index('...');
      assert.deepEqual(a.shape, [1]);
      assert.equal(a.item(), 1);
    });

    it('slice', () => {
      const a = mx.arange(64, mx.int32);
      assert.deepEqual(a.index(mx.Slice(2, 50, 8)).tolist(),
                       [2, 10, 18, 26, 34, 42]);
    });

    const a = mx.arange(64, mx.int32).reshape([8, 8]);

    it('array', () => {
      assert.deepEqual(a.index(mx.array([0, 1], mx.uint32)).tolist(),
                       [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]);
    });

    it('int', () => {
      const sliced = a.index(5);
      assert.deepEqual(sliced.tolist(), [40, 41, 42, 43, 44, 45, 46, 47]);
      assert.deepEqual(sliced.shape, [8]);
    });

    it('negative', () => {
      const sliced = a.index(-1);
      assert.deepEqual(sliced.tolist(), [56, 57, 58, 59, 60, 61, 62, 63]);
      assert.deepEqual(sliced.shape, [8]);
    });

    it('newaxis', () => {
      const sliced = a.index(null);
      assert.deepEqual(sliced.shape, [1, 8, 8]);
      assertArrayAllTrue(mx.equal(a, sliced));
    });

    it('invalidRange', () => {
      const sliced = a.index(mx.Slice(8, 3));
      assert.equal(sliced.size, 0);
    });

    it('clipPastEnd', () => {
      const sliced = a.index(mx.Slice(7, 10));
      assert.deepEqual(sliced.tolist(), [[56, 57, 58, 59, 60, 61, 62, 63]]);
    });
  });

  it('indexingGrad', () => {
    let x = mx.array([[1, 2], [3, 4]], mx.float32);
    let ind = mx.array([0, 1, 0], mx.float32);

    function indexFn(x, ind) {
      return mx.sum(x.index(ind.astype(mx.int32)));
    }

    let [gradX, gradInd] = mx.grad(indexFn, [0, 1])(x, ind);
    const expected = mx.array([[2, 2], [1, 1]]);

    assertArrayAllTrue(mx.arrayEqual(gradX, expected));
    assertArrayAllTrue(mx.arrayEqual(gradInd, mx.zeros(ind.shape)));
  });

  describe('multiDimentionalIndexing', () => {
    const a = mx.arange(64, mx.int32).reshape([8, 8]);

    it('newaxis', () => {
      const sliced = a.index(mx.Slice(), null);
      assert.deepEqual(sliced.shape, [8, 1, 8]);
      assertArrayAllTrue(mx.equal(a, sliced));
    });

    it('ints', () => {
      const sliced = a.index(0, 0);
      assert.equal(sliced.item(), 0);
      assert.equal(sliced.ndim, 0);
    });

    it('slices', () => {
      let sliced = a.index(mx.Slice(2,4), mx.Slice(5));
      assert.deepEqual(sliced.shape, [2, 3]);
      assert.deepEqual(sliced.tolist(), [[21, 22, 23], [29, 30, 31]]);

      sliced = a.index(mx.Slice(), mx.Slice(0, 5));
      assert.deepEqual(sliced.shape, [8, 5]);
    });

    it('strides', () => {
      const sliced = a.index(mx.Slice(), mx.Slice(null, null, 2));
      assert.deepEqual(sliced.shape, [8, 4]);
    });

    it('negative', () => {
      const sliced = a.index(mx.Slice(-2), mx.Slice(null, -1));
      assert.deepEqual(sliced.shape, [2, 7]);
    });

    it('intSlices', () => {
      let sliced = a.index(0, mx.Slice(null, 5));
      assert.deepEqual(sliced.shape, [5]);
      assert.deepEqual(sliced.tolist(), [0, 1, 2, 3, 4]);

      sliced = a.index(0, mx.Slice(null, -1));
      assert.deepEqual(sliced.shape, [7]);
      assert.deepEqual(sliced.tolist(), [0, 1, 2, 3, 4, 5, 6]);
    });

    const idx = mx.array([0, 1, 2, 7, 5], mx.uint32);

    it('arrayInt', () => {
      const sliced = a.index(idx, 0);
      assert.deepEqual(sliced.shape, [5]);
      assert.deepEqual(sliced.tolist(), [0, 8, 16, 56, 40]);
    });

    it('arraySlice', () => {
      const sliced1 = a.index(idx, mx.Slice(null, 5));
      assert.deepEqual(sliced1.shape, [5, 5]);
      const sliced2 = a.index(mx.Slice(null, 5), idx);
      assert.deepEqual(sliced2.shape, [5, 5]);
      assert.notDeepEqual(sliced1.tolist(), sliced2.tolist());
    });

    it('arrays', () => {
      const b = mx.arange(16).reshape([4, 4]);
      assert.deepEqual(b.index(mx.array([0, 1, 2, 3], mx.int32),
                               mx.array([0, 1, 2, 3], mx.int32)).tolist(),
                       [0, 5, 10, 15]);
      assert.deepEqual(b.index(mx.array([[0, 1]], mx.int32),
                               mx.array([[0], [1], [3]], mx.int32)).tolist(),
                       [[0, 4], [1, 5], [3, 7]]);
    });
  });

  describe('itemPut', () => {
    it('simple', () => {
      let a = mx.array(0);
      a.indexPut_(null, 1);
      assert.equal(a.item(), 1);

      a = mx.array([1, 2, 3]);
      a.indexPut_(0, 2);
      assert.deepEqual(a.tolist(), [2, 2, 3]);

      a.indexPut_(-1, 2);
      assert.deepEqual(a.tolist(), [2, 2, 2]);

      a.indexPut_(0, mx.array([[[1]]]));
      assert.deepEqual(a.tolist(), [1, 2, 2]);

      a.indexPut_(mx.Slice(), 0);
      assert.deepEqual(a.tolist(), [0, 0, 0]);

      a.indexPut_(null, 1);
      assert.deepEqual(a.tolist(), [1, 1, 1]);

      a.indexPut_(mx.Slice(0, 1), 2);
      assert.deepEqual(a.tolist(), [2, 1, 1]);

      a.indexPut_(mx.Slice(0, 2), 3);
      assert.deepEqual(a.tolist(), [3, 3, 1]);

      a.indexPut_(mx.Slice(0, 3), 4);
      assert.deepEqual(a.tolist(), [4, 4, 4]);

      a.indexPut_(mx.Slice(0, 1), mx.array(0));
      assert.deepEqual(a.tolist(), [0, 4, 4]);

      a.indexPut_(mx.Slice(0, 1), mx.array([1]));
      assert.deepEqual(a.tolist(), [1, 4, 4]);

      assert.throws(() => {
        a.indexPut_(mx.Slice(0, 1), mx.array([2, 3]));
      }, Error);

      a.indexPut_(mx.Slice(0, 2), mx.array([2, 2]));
      assert.deepEqual(a.tolist(), [2, 2, 4]);

      a.indexPut_(mx.Slice(), mx.array([[[[1, 1, 1]]]]));
      assert.deepEqual(a.tolist(), [1, 1, 1]);
    });

    it('arrayValue', () => {
      let a = mx.zeros([3, 3]);
      a.indexPut_(0, 1);
      assert.deepEqual(a.tolist(), [[1, 1, 1], [0, 0, 0], [0, 0, 0]]);

      a = mx.zeros([3, 3]);
      a.indexPut_(-1, 1);
      assert.deepEqual(a.tolist(), [[0, 0, 0], [0, 0, 0], [1, 1, 1]]);

      a = mx.zeros([3, 3]);
      a.indexPut_(mx.Slice(0, 2), 1);
      assert.deepEqual(a.tolist(), [[1, 1, 1], [1, 1, 1], [0, 0, 0]]);

      a = mx.zeros([3, 3]);
      a.indexPut_(mx.Slice(0, 2), [[0, 1, 2], [3, 4, 5]]);
      assert.deepEqual(a.tolist(), [[0, 1, 2], [3, 4, 5], [0, 0, 0]]);

      assert.throws(() => {
        a = mx.array(0);
        a.indexPut_(0, mx.array(1));
      }, Error);
    });

    it('arrayIndex', () => {
      let a = mx.zeros([3, 3]);
      a.indexPut_(mx.array([0, 1, 2], mx.uint32), 1);
      assert.deepEqual(a.tolist(), [[1, 1, 1], [1, 1, 1], [1, 1, 1]]);

      a = mx.zeros([3, 3]);
      a.indexPut_(mx.array([0, 1, 2], mx.uint32), mx.array(3));
      assert.deepEqual(a.tolist(), [[3, 3, 3], [3, 3, 3], [3, 3, 3]]);

      a = mx.zeros([3, 3]);
      a.indexPut_(mx.array([0, 1, 2], mx.uint32), mx.array([3]));
      assert.deepEqual(a.tolist(), [[3, 3, 3], [3, 3, 3], [3, 3, 3]]);

      a = mx.zeros([3, 3]);
      a.indexPut_(mx.array([0, 1], mx.uint32), mx.array([3]));
      assert.deepEqual(a.tolist(), [[3, 3, 3], [3, 3, 3], [0, 0, 0]]);

      a = mx.zeros([3, 2]);
      a.indexPut_(mx.array([0, 1], mx.uint32), mx.array([[3, 3], [4, 4]]));
      assert.deepEqual(a.tolist(), [[3, 3], [4, 4], [0, 0]]);

      a = mx.zeros([3, 2]);
      a.indexPut_(mx.array([0, 0, 1], mx.uint32), mx.array([[3, 3], [4, 4], [5, 5]]));
      assert.deepEqual(a.tolist(), [[4, 4], [5, 5], [0, 0]]);
    });

    it('nullSlices', () => {
      let a = mx.array(0);
      a.indexPut_([null, null], 1);
      assert.equal(a.item(), 1);

      a.indexPut_([null, null], mx.array(2));
      assert.equal(a.item(), 2);

      a.indexPut_([null, null], mx.array([[[3]]]));
      assert.equal(a.item(), 3);

      a.indexPut_([], 4);
      assert.equal(a.item(), 4);
    });

    it('multipleSlices', () => {
      let a = mx.zeros([3, 3]);
      a.indexPut_([mx.array([0, 2], mx.uint32), mx.Slice(1, 2)], 1);
      assert.deepEqual(a.tolist(), [[0, 1, 0], [0, 0, 0], [0, 1, 0]]);

      a = mx.zeros(5);
      a.indexPut_([null, null, mx.array([2, 3], mx.uint32)], mx.arange(2));
      assert.deepEqual(a.tolist(), [0, 0, 0, 1, 0]);

      assert.throws(() => {
        a = mx.array([4, 3, 4]);
        a.indexPut_([mx.array([2, 3], mx.uint32), null, mx.array([2, 3], mx.uint32)], mx.arange(2));
      }, Error);

      a = mx.zeros([3, 3]);
      a.indexPut_([mx.Slice(0, 2), mx.Slice(0, 2)], 1);
      assert.deepEqual(a.tolist(), [[1, 1, 0], [1, 1, 0], [0, 0, 0]]);

      a = mx.zeros([3, 3]);
      a.indexPut_([mx.Slice(0, 2), mx.Slice(0, 2)], mx.arange(2));
      assert.deepEqual(a.tolist(), [[0, 1, 0], [0, 1, 0], [0, 0, 0]]);

      a = mx.zeros([3, 3]);
      a.indexPut_([mx.Slice(0, 2), mx.Slice(0, 2)], mx.arange(2).reshape([2, 1]));
      assert.deepEqual(a.tolist(), [[0, 0, 0], [1, 1, 0], [0, 0, 0]]);

      a = mx.zeros([3, 3]);
      a.indexPut_([mx.Slice(0, 2), mx.Slice(0, 2)], mx.arange(4).reshape([2, 2]));
      assert.deepEqual(a.tolist(), [[0, 1, 0], [2, 3, 0], [0, 0, 0]]);

      assert.throws(() => {
        a = mx.zeros([2, 2, 2]);
        a.indexPut_(['...', '...'], 1);
      }, Error);

      assert.throws(() => {
        a = mx.zeros([2, 2, 2, 2, 2]);
        a.indexPut_([0, '...', 0, '...', 0], 1);
      }, Error);

      assert.throws(() => {
        a = mx.zeros([2, 2]);
        a.indexPut_([0, 0, 0], 1);
      }, Error);

      a = mx.zeros([2, 2, 2, 2]);
      a.indexPut_([null, '...', null], 1);
      assert.deepEqual(a.tolist(), mx.ones([2, 2, 2, 2]).tolist());

      a = mx.zeros([2, 3, 4, 5, 3]);
      a.indexPut_(['...', 0], 1);
      assert.deepEqual(a.index('...', 0).tolist(), mx.ones([2, 3, 4, 5]).tolist());

      a = mx.zeros([2, 3, 4, 5, 3]);
      a.indexPut_([mx.Slice(), 0], 1);
      assert.deepEqual(a.index(mx.Slice(), 0).tolist(), mx.ones([2, 4, 5, 3]).tolist());

      a = mx.zeros([2, 2, 2, 2, 2, 2]);
      a.indexPut_([0, 0], 1);
      assert.deepEqual(a.index(0, 0).tolist(), mx.ones([2, 2, 2, 2]).tolist());
    });
  });

  it('arrayAt', function() {
    // FIXME(zcbenz): This test timeouts on CI.
    if (process.env.CI == 'true' &&
        process.platform == 'darwin' &&
        process.arch == 'arm64') {
      this.skip();
    }

    let a = mx.array(1);
    a = a.at(null).add(1);
    assert.equal(a.item(), 2);

    a = mx.array([0, 1, 2]);
    a = a.at(1).add(2);
    assert.deepEqual(a.tolist(), [0, 3, 2]);

    a = a.at(mx.array([0, 0, 0, 0], mx.int32)).add(1);
    assert.deepEqual(a.tolist(), [4, 3, 2]);

    a = mx.zeros([10, 10]);
    a = a.at(0).add(mx.arange(10));
    assert.deepEqual(a.index(0).tolist(), [...Array(10).keys()]);

    a = mx.zeros([10, 10]);
    const indexX = mx.array([0, 2, 3, 7], mx.int32);
    const indexY = mx.array([3, 3, 1, 2], mx.int32);
    const u = mx.random.uniform(0, 1, [4]);
    a = a.at(indexX, indexY).add(u);
    assertArrayAllTrue(mx.allclose(a.sum(), u.sum()));
    assertArrayAllTrue(mx.allclose(a.sum(), u.sum()));
    assert.deepEqual(a.index(indexX, indexY).tolist(), u.tolist());

    const index = [mx.array([0, 4], mx.int32), mx.Slice(), 0];
    a = mx.random.uniform(0, 1, [10, 5, 2]);
    a.indexPut_(index, 0);
    let update = mx.ones([2, 5]);
    a = a.at(...index).add(update);
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), update));
    a = a.at(...index).subtract(update);
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), mx.zerosLike(update)));
    a = a.at(...index).add(mx.multiply(update, 2));
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), mx.multiply(update, 2)));
    a = a.at(...index).multiply(mx.multiply(update, 2));
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), mx.multiply(update, 4)));
    a = a.at(...index).divide(mx.multiply(update, 3));
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), mx.multiply(update, 4 / 3)));

    update = mx.arange(10).reshape(2, 5);
    a.indexPut_(index, 5);
    a = a.at(...index).maximum(update);
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), mx.maximum(a.index(...index), update)));
    a.indexPut_(index, 5);
    a = a.at(...index).minimum(update);
    assertArrayAllTrue(mx.arrayEqual(a.index(...index), mx.minimum(a.index(...index), update)));

    update = mx.array([1, 2]).index(null, null, null);
    let src = mx.array([1, 2]).index(null, mx.Slice());
    src = src.at(mx.Slice(0, 1)).add(update);
    assertArrayAllTrue(mx.arrayEqual(src, [[2, 4]]));
  });

  it('sliceNegativeStep', () => {
    let a = mx.arange(20);

    let b = a.index(mx.Slice(null, null, -1));
    assert.deepEqual(b.tolist(), mx.arange(19, -1, -1).tolist());

    b = a.index(mx.Slice(-3, 3, -1));
    assert.deepEqual(b.tolist(), mx.arange(17, 3, -1).tolist());

    b = a.index(mx.Slice(25, -50, -1));
    assert.deepEqual(b.tolist(), mx.arange(19, -1, -1).tolist());

    b = a.index(mx.Slice(null, null, -3));
    assert.deepEqual(b.tolist(), [...Array(7).keys()].map(i => 19 - 3 * i));

    b = a.index(mx.Slice(-3, 3, -3));
    assert.deepEqual(b.tolist(), [17, 14, 11, 8, 5]);

    b = a.index(mx.Slice(25, -50, -3));
    assert.deepEqual(b.tolist(), [19, 16, 13, 10, 7, 4, 1]);

    b = a.index(mx.Slice(0, 20, -3));
    assert.deepEqual(b.tolist(), []);

    a = mx.arange(3 * 6 * 4).reshape([3, 6, 4]);

    b = a.index('...', mx.Slice(null, null, -1));
    assert.deepEqual(b.tolist(),
                     [...Array(3)].map((_,i) => [...Array(6)].map((_,j) => [...Array(4)].map((_,k) => 4 * (6 * i + j) + 3 - k))));
  });

  it('deepGraphs', function() {
    this.timeout(10_000);
    this.slow(5_000);
    // The following tests should simply run cleanly without a segfault or
    // crash due to exceeding recursion depth limits.

    // Deep graph destroyed without eval.
    let x = mx.array([1.0, 2.0]);
    for (let i = 0; i < 100_000; i++) {
      x = mx.sin(x);
    }

    // Duplicate input deep graph destroyed without eval.
    x = mx.array([1.0, 2.0]);
    for (let i = 0; i < 100_000; i++) {
      x = mx.add(x, x);
    }

    // Deep graph with siblings destroyed without eval.
    x = mx.array([1, 2]);
    for (let i = 0; i < 100_000; i++) {
      x = mx.concat(mx.split(x, 2));
    }

    // Deep graph with eval.
    x = mx.array([1.0, 2.0]);
    for (let i = 0; i < 100_000; i++) {
      x = mx.sin(x);
    }
    mx.eval(x);
  });
});
