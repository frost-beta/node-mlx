import {core as mx, nn} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('init', () => {
  it('constant', () => {
    const value = 5.0;
    [mx.float32, mx.float16].forEach(dtype => {
      const initializer = nn.init.constant(value, dtype);
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
      const initializer = nn.init.normal(mean, std, dtype);
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
      const initializer = nn.init.uniform(low, high, dtype);
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
      const initializer = nn.init.identity(dtype);
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
        const initializer = nn.init[name](dtype);
        [[3, 3], [3, 3, 3]].forEach(shape => {
          const result = initializer(mx.array(mx.zeros(shape)));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype);
        });
      });
    });
  });

  it('sparse', () => {
    const mean = 0.0;
    const std = 1.0;
    const sparsity = 0.5;
    const dtypes: mx.Dtype[] = [mx.float32, mx.float16];
    dtypes.forEach(dtype => {
      const initializer = nn.init.sparse(sparsity, mean, std, dtype);
      const shapes: [number, number][] = [[3, 2], [2, 2], [4, 3]];
      shapes.forEach(shape => {
        let result = initializer(mx.array(mx.zeros(shape), dtype));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype);
        assertArrayAllTrue(
          mx.greaterEqual(mx.sum(mx.equal(result, 0)), 0.5 * shape[0] * shape[1])
        );
      });
      assert.throws(() => {
        initializer(mx.zeros([1], dtype));
      }, Error);
    });
  });
});
