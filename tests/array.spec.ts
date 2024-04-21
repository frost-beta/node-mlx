import mx from '..';
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

describe('array', () => {
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

      x = mx.array([1, 2, 3], mx.int32);
      assert.equal(x.dtype, mx.int32);
      assert.deepEqual(x.tolist(), [1, 2, 3]);
    });
  });
});

function assertArray(a: mx.array, assertion: (arrays: boolean[]) => void) {
  assert.isTrue(a instanceof mx.array);
  assert.equal(a.dtype, mx.bool_);
  if (a.ndim == 0) {
    assertion([ a.item() as boolean ]);
  } else {
    const list = a.tolist();
    assertion(list as boolean[]);
  }
}

const assertArrayAllTrue = (a) => assertArray(a, (arrays) => assert.notInclude(arrays, false));
const assertArrayAllFalse = (a) => assertArray(a, (arrays) => assert.notInclude(arrays, true));
