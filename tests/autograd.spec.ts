import mx from '..';
import {assert} from 'chai';

describe('autograd', () => {
  it('jvp', () => {
    const fun1 = (x) => mx.multiply(2, x);
    const [out, dout] = mx.jvp(fun1, [mx.array(1.0)], [mx.array(2.0)]);
    assert.equal(out[0].item(), 2.0);
    assert.equal(dout[0].item(), 4.0);

    const fun2 = (x, y) => mx.multiply(x, y);
    let result = mx.jvp(fun2, [mx.array(4.0), mx.array(2.0)], [mx.array(3.0), mx.array(2.0)]);
    const out2 = result[1];
    assert.equal(out2[0].item(), 4.0 * 2.0 + 2.0 * 3.0);

    const fun3 = (x, y, z) => [mx.multiply(x, y), mx.multiply(y, z)];
    result = mx.jvp(
      fun3,
      [mx.array(2.0), mx.array(4.0), mx.array(6.0)],
      [mx.array(1.0), mx.array(3.0), mx.array(1.0)],
    );
    const out3 = result[1];
    assert.equal(out3.length, 2);
    assert.equal(out3[0].item(), 4.0 * 1.0 + 2.0 * 3.0);
    assert.equal(out3[1].item(), 4.0 * 1.0 + 6.0 * 3.0);
  });

  it('vjp', () => {
    const fun1 = (x) => mx.multiply(2, x);
    const [out, dout] = mx.vjp(fun1, [mx.array(1.0)], [mx.array(2.0)]);
    assert.equal(out[0].item(), 2.0);
    assert.equal(dout[0].item(), 4.0);

    const fun2 = (x, y) => mx.multiply(x, y);
    let result = mx.vjp(fun2, [mx.array(4.0), mx.array(2.0)], [mx.array(3.0)]);
    const dout2 = result[1];
    assert.equal(dout2[0].item(), 6.0);
    assert.equal(dout2[1].item(), 12.0);

    const fun3 = (x, y, z) => [mx.multiply(x, y), mx.multiply(y, z)];
    result = mx.vjp(
      fun3,
      [mx.array(2.0), mx.array(4.0), mx.array(6.0)],
      [mx.array(1.0), mx.array(3.0)],
    );
    const out3 = result[1];
    assert.equal(out3.length, 3);
    assert.equal(out3[0].item(), 4.0 * 1.0);
    assert.equal(out3[1].item(), 2.0 * 1.0 + 6.0 * 3.0);
    assert.equal(out3[2].item(), 4.0 * 3.0);
  });
});
