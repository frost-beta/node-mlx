import {core as mx, nn} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('init', () => {
  it('constant', () => {
    const value = 5.0;
    [mx.float32, mx.float16].forEach(dtype => {
      const initializer = nn.constant(value, dtype);
      [[3], [3, 3], [3, 3, 3]].forEach(shape => {
        const result = initializer(mx.array(mx.zeros(shape)));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype);
      });
    });
  });

  it('normal', () => {
    const mean = 0.0;
    const std = 1.0;
    [mx.float32, mx.float16].forEach(dtype => {
      const initializer = nn.normal(mean, std, dtype);
      [[3], [3, 3], [3, 3, 3]].forEach(shape => {
        const result = initializer(mx.array(mx.zeros(shape)));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype);
      });
    });
  });

  it('uniform', () => {
    const low = -1.0;
    const high = 1.0;
    [mx.float32, mx.float16].forEach(dtype => {
      const initializer = nn.uniform(low, high, dtype);
      [[3], [3, 3], [3, 3, 3]].forEach(shape => {
        const result = initializer(mx.array(mx.zeros(shape)));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype);
        assertArrayAllTrue(mx.all(mx.greaterEqual(result, low)));
        assertArrayAllTrue(mx.all(mx.lessEqual(result, high)));
      });
    });
  });

  it('identity', () => {
    [mx.float32, mx.float16].forEach(dtype => {
      const initializer = nn.identity(dtype);
      [[3], [3, 3], [3, 3, 3]].forEach(() => {
        let result = initializer(mx.zeros([3, 3]));
        assert(mx.arrayEqual(result, mx.eye(3)));
        assert.equal(result.dtype, dtype);
        assert.throws(() => {
          result = initializer(mx.zeros([3, 2]));
        });
      });
    });
  });

  ['glorotNormal', 'glorotUniform', 'heNormal', 'heUniform'].forEach(name => {
    it(name, () => {
      [mx.float32, mx.float16].forEach(dtype => {
        const initializer = nn[name](dtype);
        [[3, 3], [3, 3, 3]].forEach(shape => {
          const result = initializer(mx.array(mx.zeros(shape)));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype);
        });
      });
    });
  });
});
