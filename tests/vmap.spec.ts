import {core as mx} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('vmap', () => {
  it('basics', () => {
    const exp = (x: mx.array) => mx.exp(x);
    assert.throws(() => {
      mx.vmap(exp)(mx.array(1.0));
    }, Error);
    assert.throws(() => {
      mx.vmap(exp, 2)(mx.array([0, 1]));
    }, Error);
    assert.throws(() => {
      mx.vmap(exp, null, 2)(mx.array([0, 1]));
    }, Error);
  });

  describe('unary', () => {
    const ops = [
      mx.abs,
      mx.cos,
      mx.erf,
      mx.erfinv,
      mx.exp,
      mx.log,
      mx.log1p,
      mx.log2,
      mx.log10,
      mx.logicalNot,
      mx.negative,
      mx.reciprocal,
      mx.rsqrt,
      mx.sigmoid,
      mx.sign,
      mx.sin,
      mx.sqrt,
      mx.square,
      mx.degrees,
      mx.radians,
    ];

    for (const op of ops) {
      let x = mx.arange(5);
      let y = mx.vmap((x: mx.array) => op(x))(x);
      assertArrayAllTrue(mx.arrayEqual(y, op(x), true));

      x = mx.arange(8).reshape([2, 4]);
      y = mx.vmap((x: mx.array) => op(x))(x);
      assertArrayAllTrue(mx.arrayEqual(y, op(x), true));

      y = mx.vmap((x: mx.array) => op(x), 1, 1)(x);
      assertArrayAllTrue(mx.arrayEqual(y, op(x), true));
    };
  });

  it('binary', () => {
    const ops = [
      mx.add,
      mx.divide,
      mx.equal,
      mx.greater,
      mx.greaterEqual,
      mx.less,
      mx.lessEqual,
      mx.logaddexp,
      mx.maximum,
      mx.minimum,
      mx.multiply,
      mx.power,
      mx.subtract,
      mx.logicalOr,
      mx.logicalAnd,
    ];
    for (const op of ops) {
      let x = mx.random.uniform(0, 1, [5]);
      let y = mx.random.uniform(0, 1, [5]);
      let out = mx.vmap((a: mx.array, b: mx.array) => op(a, b))(x, y);
      assertArrayAllTrue(mx.arrayEqual(out, op(x, y)));

      x = mx.random.uniform(0, 1, [2, 4]);
      y = mx.random.uniform(0, 1, [2, 4]);
      out = mx.vmap((a: mx.array, b: mx.array) => op(a, b))(x, y);
      assertArrayAllTrue(mx.arrayEqual(out, op(x, y)));

      out = mx.vmap((a: mx.array, b: mx.array) => op(a, b), [0, 0], 0)(x, y);
      assertArrayAllTrue(mx.arrayEqual(out, op(x, y)));

      y = mx.random.uniform(0, 1, [4, 2]);
      out = mx.vmap((a: mx.array, b: mx.array) => op(a, b), [0, 1], 0)(x, y);
      assertArrayAllTrue(mx.arrayEqual(out, op(x, y.T)));

      out = mx.vmap((a: mx.array, b: mx.array) => op(a, b), [0, 1], 1)(x, y);
      assertArrayAllTrue(mx.arrayEqual(out, op(x, y.T).T));
    }
  });

  it('vmapIndexing', () => {
    const x = mx.arange(16).reshape([2, 2, 2, 2]);
    const inds = mx.array([[0, 1, 0], [1, 1, 0]], mx.int32);

    let out = mx.vmap((x: mx.array, y: mx.array) => x.index(y), [0, 0])(x, inds);
    const expected = mx.array(
      [
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
        [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
      ]
    );
    assert(mx.arrayEqual(out, expected));

    out = mx.vmap((x: mx.array, y: mx.array) => x.index(y), [0, null])(x, inds);
    const expected2 = mx.array(
      [
        [
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
          [[[4, 5], [6, 7]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
        ],
        [
          [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
          [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
        ],
      ]
    );
    assert(mx.arrayEqual(out, expected2));

    out = mx.vmap((x: mx.array, y: mx.array) => x.index(y), [null, 0])(x, inds);
    const expected3 = mx.array(
      [
        [
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
          [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        ],
        [
          [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
          [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        ],
      ]
    );
    assert(mx.arrayEqual(out, expected3));

    const inds2 = mx.array([[0, 1, 0], [0, 1, 0]], mx.int32);
    out = mx.vmap((x: mx.array, y: mx.array, z: mx.array) => x.index(y, z), [null, 0, 0])(x, inds, inds2);
    const expected4 = mx.array(
      [
        [[[0, 1], [2, 3]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
        [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
      ]
    );
    assert(mx.arrayEqual(out, expected4));
  });

  it('vmapReduce', () => {
    let a = mx.ones([5, 5], mx.int32);
    let out = mx.vmap((x: mx.array) => x.sum())(a);
    assertArrayAllTrue(mx.arrayEqual(out, mx.full([5], 5)));

    out = mx.vmap((x: mx.array) => x.sum(null, true))(a);
    assertArrayAllTrue(mx.arrayEqual(out, mx.full([5, 1], 5)));

    out = mx.vmap((x: mx.array) => x.sum(0))(a);
    assertArrayAllTrue(mx.arrayEqual(out, mx.full([5], 5)));

    a = mx.ones([5, 3, 2], mx.int32);
    out = mx.vmap((x: mx.array) => x.sum([0, 1]))(a);
    assertArrayAllTrue(mx.arrayEqual(out, mx.full([5], 6)));

    a = mx.ones([5, 3, 2], mx.int32);
    out = mx.vmap((x: mx.array) => x.sum([0, 1]), [1])(a);
    assertArrayAllTrue(mx.arrayEqual(out, mx.full([3], 10)));

    a = mx.ones([5, 3, 2], mx.int32);
    out = mx.vmap((x: mx.array) => x.sum([0, 1]), [2])(a);
    assertArrayAllTrue(mx.arrayEqual(out, mx.full([2], 15)));
  });

  it('vmapArgreduce', () => {
    const a = mx.array([[1, 2, 3], [2, 3, 1]]);
    let out = mx.vmap((x: mx.array) => mx.argmin(x))(a);
    let expected = mx.array([0, 2]);
    assertArrayAllTrue(mx.arrayEqual(out, expected));

    out = mx.vmap((x: mx.array) => mx.argmax(x))(a);
    expected = mx.array([2, 1]);
    assertArrayAllTrue(mx.arrayEqual(out, expected));
  });

  it('vmapMean', () => {
    let a = mx.reshape(mx.arange(8), [2, 4]);
    let out = mx.vmap((x: mx.array) => mx.mean(x))(a);
    let expected = mx.mean(a, 1);
    assertArrayAllTrue(mx.allclose(out, expected));

    a = mx.reshape(mx.arange(16), [2, 2, 4]);
    out = mx.vmap(mx.vmap((x: mx.array) => mx.mean(x)))(a);
    expected = mx.mean(a, 2);
    assertArrayAllTrue(mx.allclose(out, expected));
  });

  it('mismatchInputSizes', () => {
    const a = mx.ones([10, 1]);
    let b = mx.ones([1, 1, 1, 5]);
    assert.throws(() => {
      let out = mx.vmap((x: mx.array, y: mx.array) => mx.add(x, y))(a, b);
    }, Error);

    b = mx.ones([10, 5]);
    assert.throws(() => {
      let out = mx.vmap((x: mx.array, y: mx.array) => mx.add(x, y), [0, 1])(a, b);
    }, Error);
  });

  it('vmapMatmul', () => {
    let a = mx.random.uniform(0, 1, [2, 3, 4]);
    let b = mx.random.uniform(0, 1, [4, 3]);

    let out = mx.vmap((a: mx.array, b: mx.array) => mx.matmul(a, b), [0, -1])(a, b);
    assertArrayAllTrue(mx.allclose(out, mx.matmul(a, b)));

    let c = mx.random.uniform(0, 1, [3]);
    out = mx.vmap((c: mx.array, a: mx.array, b: mx.array) => mx.addmm(c, a, b), [-1, 0, -1])(c, a, b);
    assertArrayAllTrue(mx.allclose(out, mx.addmm(c, a, b)));

    b = mx.random.uniform(0, 1, [4, 2]);
    out = mx.vmap((a: mx.array, b: mx.array) => mx.matmul(a, b), [1, -1], 1)(a, b);
    let expected = mx.moveaxis(mx.matmul(mx.moveaxis(a, 1, 0), b), 0, 1);
    assertArrayAllTrue(mx.allclose(out, expected));

    c = mx.random.uniform(0, 1, [2]);
    out = mx.vmap((c: mx.array, a: mx.array, b: mx.array) => mx.addmm(c, a, b), [-1, 1, -1])(c, a, b);
    assertArrayAllTrue(mx.allclose(out, mx.addmm(c, mx.moveaxis(a, 1, 0), b)));

    a = mx.random.uniform(0, 1, [2, 3, 4]);
    b = mx.random.uniform(0, 1, [4, 2, 3]);
    out = mx.vmap((a: mx.array, b: mx.array) => mx.matmul(a, b), [0, 1])(a, b);
    expected = mx.matmul(a, mx.moveaxis(b, 1, 0));
    assertArrayAllTrue(mx.allclose(out, expected));

    c = mx.random.uniform(0, 1, [3, 3, 2]);
    out = mx.vmap((c: mx.array, a: mx.array, b: mx.array) => mx.addmm(c, a, b), [2, 0, 1])(c, a, b);
    expected = mx.addmm(mx.moveaxis(c, 2, 0), a, mx.moveaxis(b, 1, 0));
    assertArrayAllTrue(mx.allclose(out, expected));
  });

  it('vmapSvd', () => {
    const a = mx.random.uniform(0, 1, [3, 4, 2]);
    const cpuSvd = (x: mx.array) => mx.linalg.svd(x, mx.cpu);

    let [Us, Ss, Vts] = mx.vmap(cpuSvd, 0)(a);
    assert.deepEqual(Us.shape, [a.shape[0], a.shape[1], a.shape[1]]);
    assert.deepEqual(Ss.shape, [a.shape[0], a.shape[2]]);
    assert.deepEqual(Vts.shape, [a.shape[0], a.shape[2], a.shape[2]]);

    for (let i = 0; i < a.shape[0]; i++) {
      const M = a.index(i);
      const U = Us.index(i);
      const S = Ss.index(i);
      const Vt = Vts.index(i);
      assertArrayAllTrue(
          mx.allclose(mx.matmul(mx.matmul(U.index(mx.Slice(), mx.Slice(null, S.length)), mx.diag(S)), Vt),
                      M, 1e-5, 1e-7));
    }

    [Us, Ss, Vts] = mx.vmap(cpuSvd, 1)(a);
    assert.deepEqual(Us.shape, [a.shape[1], a.shape[0], a.shape[0]]);
    assert.deepEqual(Ss.shape, [a.shape[1], a.shape[2]]);
    assert.deepEqual(Vts.shape, [a.shape[1], a.shape[2], a.shape[2]]);

    for (let i = 0; i < a.shape[1]; i++) {
      const M = a.index(mx.Slice(), i, mx.Slice());
      const U = Us.index(i);
      const S = Ss.index(i);
      const Vt = Vts.index(i);
      assertArrayAllTrue(
          mx.allclose(mx.matmul(mx.matmul(U.index(mx.Slice(), mx.Slice(null, S.length)), mx.diag(S)), Vt),
                      M, 1e-5, 1e-7));
    }
  });

  it('vmapInverse', () => {
    let a = mx.random.uniform(0, 1, [3, 4, 4]);
    const cpuInv = (x: mx.array) => mx.linalg.inv(x, mx.cpu);

    let invs = mx.vmap(cpuInv, 0)(a);
    for (let i = 0; i < a.shape[0]; i++) {
      assertArrayAllTrue(
          mx.allclose(mx.matmul(a.index(i), invs.index(i)),
                      mx.eye(a.shape[1]), 0, 1e-5));
    }

    a = mx.random.uniform(0, 1, [4, 3, 4]);
    assert.throws(() => {
      mx.eval(cpuInv(a));
    }, Error);

    invs = mx.vmap(cpuInv, [1])(a);
    for (let i = 0; i < a.shape[1]; i++) {
      assertArrayAllTrue(
          mx.allclose(mx.matmul(a.index(mx.Slice(), i, mx.Slice()), invs.index(i)),
                      mx.eye(a.shape[0]), 0, 1e-5));
    }
  });
});
