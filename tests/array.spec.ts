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
      assert.isTrue(mx.all(mx.equal(a, b)));
      assert.isFalse(mx.all(mx.equal(a, c)));
    });

    it('arrayEqScalar', () => {
      const a = mx.array([1, 2, 3]);
      const b = 1;
      const c = 4;
      const d = 2.5;
      const e = mx.array([1, 2.5, 3.25]);
      assert.isTrue(mx.any(mx.equal(a, b)));
      assert.isFalse(mx.all(mx.equal(a, c)));
      assert.isFalse(mx.all(mx.equal(a, d)));
      assert.isTrue(mx.any(mx.equal(a, e)));
    });

    it('listEqualsArray', () => {
      const a = mx.array([1, 2, 3]);
      const b = [1, 2, 3];
      const c = [1, 2, 4];
      assert.isFalse(mx.equal(a, b));
      assert.isFalse(mx.equal(a, c));
    });
  });
});
