import {core as mx} from '..';
import {assertArrayAllTrue, assertArrayNotAllTrue, assertArrayAllFalse} from './utils';
import {assert} from 'chai';

describe('random', () => {
  it('globalRng', () => {
    mx.random.seed(3);
    const a = mx.random.uniform(0, 1);
    const b = mx.random.uniform(0, 1);

    mx.random.seed(3);
    const x = mx.random.uniform(0, 1);
    const y = mx.random.uniform(0, 1);

    assert.equal(a.item(), x.item());
    assert.equal(y.item(), b.item());
  });

  it('key', () => {
    const k1 = mx.random.key(0);
    let k2 = mx.random.key(0);
    assertArrayAllTrue(mx.arrayEqual(k1, k2));

    k2 = mx.random.key(1);
    assertArrayAllFalse(mx.arrayEqual(k1, k2));
  });

  it('uniform', () => {
    let key = mx.random.key(0);
    let a = mx.random.uniform(0, 1, [], mx.float32, key);
    assert.equal(a.shape.length, 0);
    assert.equal(a.dtype, mx.float32);

    let b = mx.random.uniform(0, 1, [], mx.float32, key);
    assert.equal(a.item(), b.item());

    a = mx.random.uniform(0, 1, [2, 3]);
    assert.deepEqual(a.shape, [2, 3]);

    a = mx.random.uniform(-1, 5, [1000]);
    assertArrayAllTrue(mx.less(mx.greater(a, -1), 5));

    a = mx.random.uniform(mx.array(-1), 5, [1000]);
    assertArrayAllTrue(mx.less(mx.greater(a, -1), 5));

    a = mx.random.uniform(-0.1, 0.1, [1], mx.bfloat16);
    assert.equal(a.dtype, mx.bfloat16);
  });

  it('normalAndLaplace', () => {
    for (const distributionSampler of [mx.random.normal, mx.random.laplace]) {
      let key = mx.random.key(0);
      let a = distributionSampler([], mx.float32, 0, 1, key);
      assert.equal(a.shape.length, 0);
      assert.equal(a.dtype, mx.float32);

      let b = distributionSampler([], mx.float32, 0, 1, key);
      assert.equal(a.item(), b.item());

      a = distributionSampler([2, 3]);
      assert.deepEqual(a.shape, [2, 3]);

      [mx.float16, mx.bfloat16].forEach(t => {
        a = distributionSampler([], t);
        assert.equal(a.dtype, t);
      });

      const loc = 1.0;
      const scale = 2.0;

      a = distributionSampler([3, 2], mx.float32, loc, scale, key);
      b = mx.add(mx.multiply(scale, distributionSampler([3, 2], mx.float32, 0, 1, key)), loc);
      assertArrayAllTrue(mx.allclose(a, b));

      a = distributionSampler([3, 2], mx.float16, loc, scale, key);
      b = mx.add(mx.multiply(scale, distributionSampler([3, 2], mx.float16, 0, 1, key)), loc);
      assertArrayAllTrue(mx.allclose(a, b));

      assert.equal(distributionSampler().dtype, distributionSampler().dtype);

      [mx.float16, mx.bfloat16].forEach(hp => {
        a = mx.abs(distributionSampler([10000], hp, 0, 1));
        assertArrayAllTrue(mx.less(a, Infinity));
      });
    }
  });

  it('multivariateNormal', () => {
    let key = mx.random.key(0);
    let mean = mx.array([0, 0]);
    let cov = mx.array([[1, 0], [0, 1]]);

    let a = mx.random.multivariateNormal(mean, cov, [], mx.float32, key, mx.cpu);
    assert.deepEqual(a.shape, [2]);

    let floatTypes = [mx.float32];
    for (const t of floatTypes) {
      a = mx.random.multivariateNormal(mean, cov, [], t, key, mx.cpu);
      assert.equal(a.dtype, t);
    }
    let otherTypes = [mx.int8, mx.int32, mx.int64, mx.uint8, mx.uint32, mx.uint64, mx.float16, mx.bfloat16];
    for (const t of otherTypes) {
      assert.throws(() => {
        mx.random.multivariateNormal(mean, cov, [], t, key, mx.cpu);
      }, Error);
    }

    mean = mx.array([[0, 7], [1, 2], [3, 4]]);
    cov = mx.array([[1, 0.5], [0.5, 1]]);
    a = mx.random.multivariateNormal(mean, cov, [4, 3]);
    assert.deepEqual(a.shape, [4, 3, 2]);

    const nTest = 1e5;

    const checkJointlyGaussian = (data, mean, cov) => {
      const empiricalMean = mx.mean(data, 0, true);
      const empiricalCov = mx.divide(mx.matmul(mx.subtract(data, empiricalMean).T, mx.subtract(data, empiricalMean)), data.shape[0]);
      const N = data.shape[1];
      assertArrayAllTrue(mx.allclose(empiricalMean, mean, 0.0, 10 * Math.pow(N, 2) / Math.sqrt(nTest)));
      assertArrayAllTrue(mx.allclose(empiricalCov, cov, 0.0, 10 * Math.pow(N, 2) / Math.sqrt(nTest)));
    };

    mean = mx.array([4.0, 7.0]);
    cov = mx.array([[2, 0.5], [0.5, 1]]);
    let data = mx.random.multivariateNormal(mean, cov, [nTest], mx.float32, key, mx.cpu);

    checkJointlyGaussian(data, mean, cov);

    mean = mx.arange(3);
    cov = mx.array([[1, -1, 0.5], [-1, 1, -0.5], [0.5, -0.5, 1]]);
    data = mx.random.multivariateNormal(mean, cov, [nTest], mx.float32, key, mx.cpu);
    checkJointlyGaussian(data, mean, cov);
  });

  it('randint', function() {
    this.timeout(10 * 1000);  // slow in QEMU

    let a = mx.random.randint(0, 1, []);
    assert.equal(a.shape.length, 0);
    assert.equal(a.dtype, mx.int32);

    let shape = [88];
    let low = mx.array(3);
    let high = mx.array(15);

    const key = mx.random.key(0);
    a = mx.random.randint(low, high, shape, mx.int32, key);
    assert.deepEqual(a.shape, shape);
    assert.equal(a.dtype, mx.int32);

    let b = mx.random.randint(low, high, shape, mx.int32, key);
    assert.deepEqual(a.tolist(), b.tolist());

    shape = [3, 4];
    const lowArray = mx.array([0, 0, 0]);
    low = mx.reshape(lowArray, [3, 1]);
    const highArray = mx.array([12, 13, 14, 15]);
    high = mx.reshape(highArray, [1, 4]);

    a = mx.random.randint(low, high, shape);
    assert.deepEqual(a.shape, shape);

    a = mx.random.randint(-10, 10, [1000, 1000]);
    assertArrayAllTrue(mx.logicalAnd(mx.lessEqual(-10, a), mx.less(a, 10)));

    a = mx.random.randint(10, -10, [1000, 1000]);
    assertArrayAllTrue(mx.equal(a, 10));

    assert.equal(mx.random.randint(0, 1).dtype, mx.random.randint(0, 1).dtype);
  });

  it('bernoulli', () => {
    let a = mx.random.bernoulli();
    assert.equal(a.shape.length, 0);
    assert.equal(a.dtype, mx.bool);

    const probHalf = mx.array(0.5);
    a = mx.random.bernoulli(probHalf, [5]);
    assert.deepEqual(a.shape, [5]);

    const negProb = mx.array([2.0, -2.0]);
    a = mx.random.bernoulli(negProb);
    assert.deepEqual(a.tolist(), [true, false]);
    assert.deepEqual(a.shape, [2]);

    let p = mx.array([0.1, 0.2, 0.3]);
    mx.reshape(p, [1, 3]);
    let x = mx.random.bernoulli(p, [4, 3]);
    assert.deepEqual(x.shape, [4, 3]);

    assert.throws(() => {
      mx.random.bernoulli(p, [2]);  // Bad shape
    }, Error);

    assert.throws(() => {
      mx.random.bernoulli(mx.array(0, mx.int32), [2]);  // Bad type
    }, Error);
  });

  it('truncatedNormal', () => {
    let a = mx.random.truncatedNormal(-2.0, 2.0);
    assert.equal(a.size, 1);
    assert.equal(a.dtype, mx.float32);

    a = mx.random.truncatedNormal(mx.array([]), mx.array([]));
    assert.equal(a.dtype, mx.float32);
    assert.equal(a.size, 0);

    let lower = mx.reshape(mx.array([-2.0, 0.0]), [1, 2]);
    let upper = mx.reshape(mx.array([0.0, 1.0, 2.0]), [3, 1]);
    a = mx.random.truncatedNormal(lower, upper);

    assert.deepEqual(a.shape, [3, 2]);
    assert.isTrue(mx.all(mx.lessEqual(lower, a)).item() && mx.all(mx.lessEqual(a, upper)).item());

    a = mx.random.truncatedNormal(2.0, -2.0);
    assertArrayAllTrue(mx.equal(a, 2.0));

    a = mx.random.truncatedNormal(-3.0, 3.0, [542, 399]);
    assert.deepEqual(a.shape, [542, 399]);

    lower = mx.array([-2.0, -1.0]);
    upper = mx.array([1.0, 2.0, 3.0]);
    assert.throws(() => {
      mx.random.truncatedNormal(lower, upper);  // Bad shape
    }, Error);

    assert.equal(
      mx.random.truncatedNormal(0, 1).dtype,
      mx.random.truncatedNormal(0, 1).dtype,
    );
  });

  it('gumbel', () => {
    const samples = mx.random.gumbel([100, 100]);
    assert.deepEqual(samples.shape, [100, 100]);
    assert.equal(samples.dtype, mx.float32);
    const mean = 0.5772;
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(mx.mean(samples), mean)), 0.2));
    assert.equal(mx.random.gumbel([1, 1]).dtype, mx.random.gumbel([1, 1]).dtype);
  });

  it('categorical', () => {
    const logits = mx.zeros([10, 20]);
    assert.deepEqual(mx.random.categorical(logits, -1).shape, [10]);
    assert.deepEqual(mx.random.categorical(logits, 0).shape, [20]);
    assert.deepEqual(mx.random.categorical(logits, 1).shape, [10]);

    let out = mx.random.categorical(logits);
    assert.deepEqual(out.shape, [10]);
    assert.equal(out.dtype, mx.uint32);
    assertArrayAllTrue(mx.less(mx.max(out), 20));

    out = mx.random.categorical(logits, 0, [5, 20]);
    assert.deepEqual(out.shape, [5, 20]);
    assertArrayAllTrue(mx.less(mx.max(out), 10));

    out = mx.random.categorical(logits, 1, undefined, 7);
    assert.deepEqual(out.shape, [10, 7]);

    out = mx.random.categorical(logits, 0, undefined, 7);
    assert.deepEqual(out.shape, [20, 7]);

    assert.throws(() => {
      mx.random.categorical(logits, -1, [10, 5], 5);
    }, Error);
  });

  it('permutation', () => {
    let x = mx.random.permutation(4);
    assert.deepEqual((x.tolist() as number[]).sort(), [0, 1, 2, 3]);

    x = mx.random.permutation(mx.array([0, 1, 2, 3]));
    assert.deepEqual((x.tolist() as number[]).sort(), [0, 1, 2, 3]);

    // 2-D
    x = mx.arange(16).reshape(4, 4);
    let out = mx.sort(mx.random.permutation(x, 0), 0);
    assertArrayAllTrue(mx.arrayEqual(x, out));
    out = mx.sort(mx.random.permutation(x, 1), 1);
    assertArrayAllTrue(mx.arrayEqual(x, out));

    // Basically 0 probability this should fail.
    const sortedX = mx.arange(16384);
    x = mx.random.permutation(16384);
    assertArrayAllFalse(mx.arrayEqual(sortedX, x));

    // Preserves shape / doesn't cast input to int.
    x = mx.random.permutation(mx.array([[1]]));
    assert.deepEqual(x.shape, [1, 1]);
  });
});
