import {core as mx} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

// TODO(zcbenz): The tests were written by ChatGPT, rewrite them.
describe('linalg', () => {
  it('norm', () => {
    const input = mx.array([0, 1, 2, 3]);
    const result = mx.linalg.norm(input);
    assert.closeTo(result.item() as number, 3.741, 0.001);
  });

  it('qr', () => {
    const input = mx.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]);
    const [q, r] = mx.linalg.qr(input, mx.cpu);
    const qExpected = mx.array([[-0.8571428571428571, 0.3942857142857143, 0.33142857142857146], [-0.42857142857142855, -0.9028571428571428, -0.03428571428571429], [0.2857142857142857, -0.17142857142857143, 0.9428571428571428]]);
    const rExpected = mx.array([[-14.0, -21.0, 14.0], [0.0, -175.0, 70.0], [0.0, 0.0, -35.0]]);
    assertArrayAllTrue(mx.isclose(q, qExpected));
    assertArrayAllTrue(mx.isclose(r, rExpected));
  });

  it('svd', () => {
    const input = mx.array([[4, 11], [2, 5], [8, 17]]);
    const [u, s, v] = mx.linalg.svd(input, mx.cpu);
    const uExpected = mx.array([[-0.36620161204573334, 0.8644312940738013, 0.34586548686958506], [-0.18310080602286667, 0.43221564703690067, -0.8830292018647491], [-0.9125040301143334, -0.2577291940111404, 0.31691529099600695]]);
    const sExpected = mx.array([22.76292610168457, 0.9215062260627747]);
    const vExpected = mx.array([[-0.4009232521057129, -0.9161115884780884], [-0.9161115884780884, 0.40092331171035767]]);
    assertArrayAllTrue(mx.isclose(u, uExpected));
    assertArrayAllTrue(mx.isclose(s, sExpected));
    assertArrayAllTrue(mx.isclose(v, vExpected));
  });

  it('inv', () => {
    const input = mx.array([[4, 7], [2, 6]]);
    const result = mx.linalg.inv(input, mx.cpu);
    const expected = mx.array([[0.6, -0.7], [-0.2, 0.4]]);
    assertArrayAllTrue(mx.isclose(result, expected));
  });

  it('cholesky', () => {
    const sqrtA = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], mx.float32);
    const A = mx.divide(mx.matmul(sqrtA.T, sqrtA), 81);
    const L = mx.linalg.cholesky(A, false, mx.cpu);
    const U = mx.linalg.cholesky(A, true, mx.cpu);
    assertArrayAllTrue(mx.allclose(mx.matmul(L, L.T), A, 1e-5, 1e-7));
    assertArrayAllTrue(mx.allclose(mx.matmul(U.T, U), A, 1e-5, 1e-7));

    // Multiple matrices
    const B = mx.add(A, 1 / 9);
    const AB = mx.stack([A, B]);
    const Ls = mx.linalg.cholesky(AB, false, mx.cpu);
    for (let i = 0; i < AB.length; i++) {
      const M = AB.index(i);
      const L = Ls.index(i);
      assertArrayAllTrue(mx.allclose(mx.matmul(L, L.T), M, 1e-5, 1e-7));
    }
  });

  it('pseudoInverse', () => {
    const A = mx.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], mx.float32);
    const APlus = mx.linalg.pinv(A, mx.cpu);
    assertArrayAllTrue(mx.allclose(mx.matmul(mx.matmul(A, APlus), A), A, 0, 1e-5));

    // Multiple matrices
    const B = mx.subtract(A, 100);
    const AB = mx.stack([A, B]);
    const pinvs = mx.linalg.pinv(AB, mx.cpu);
    for (let i = 0; i < AB.length; i++) {
      const M = AB.index(i);
      const MPlus = pinvs.index(i);
      assertArrayAllTrue(mx.allclose(mx.matmul(mx.matmul(M, MPlus), M), M, 0, 1e-3));
    }
  });

  it('choleskyInv', () => {
    it('choleskyInv', () => {
      mx.random.seed(7);

      const sqrtA = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], mx.float32);
      let A = mx.divide(mx.matmul(sqrtA.T, sqrtA), 81);

      const N = 3;
      A = mx.matmul(A, A.T);

      [false, true].forEach((upper) => {
        const L = mx.linalg.cholesky(A, upper, mx.cpu);
        const AInv = mx.linalg.choleskyInv(L, upper, mx.cpu);
        assertArrayAllTrue(mx.isclose(mx.matmul(A, AInv), mx.eye(N), 1e-4));
      });

      const B = mx.add(A, mx.divide(1, 9));
      const AB = mx.stack([A, B]);
      [false, true].forEach((upper) => {
        const Ls = mx.linalg.cholesky(AB, upper, mx.cpu);
        const ABInv = mx.linalg.choleskyInv(Ls, upper, mx.cpu);
        for (let i = 0; i < AB.size; ++i) {
          const M = AB.index(i);
          const MInv = ABInv.index(i);
          assertArrayAllTrue(mx.allclose(mx.matmul(M, MInv), mx.eye(N), undefined, 1e-4));
        }
      });
    });
  });

  it('crossProduct', () => {
    let a = mx.array([1.0, 2.0, 3.0]);
    let b = mx.array([4.0, 5.0, 6.0]);
    let result = mx.linalg.cross(a, b);
    assert.deepEqual(result.tolist(), [-3.0, 6.0, -3.0]);

    // Test with negative values.
    a = mx.array([-1.0, -2.0, -3.0]);
    b = mx.array([4.0, -5.0, 6.0]);
    result = mx.linalg.cross(a, b);
    assert.deepEqual(result.tolist(), [-27, -6, 13]);

    // Test with integer values.
    a = mx.array([1, 2, 3]);
    b = mx.array([4, 5, 6]);
    result = mx.linalg.cross(a, b);
    assert.deepEqual(result.tolist(), [-3, 6, -3]);

    // Test with 2D arrays and axis parameter.
    a = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    b = mx.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]);
    result = mx.linalg.cross(a, b, 1);
    assert.deepEqual(result.tolist(), [[-3.0, 6.0, -3.0], [3.0, -6.0, 3.0]]);

    // Test with incorrect vector size (should raise an exception).
    a = mx.array([1.0]);
    b = mx.array([4.0]);
    assert.throws(() => mx.linalg.cross(a, b), Error);
  });
});
