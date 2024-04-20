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
