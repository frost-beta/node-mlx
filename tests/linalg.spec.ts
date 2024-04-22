import mx from '..';
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
});
