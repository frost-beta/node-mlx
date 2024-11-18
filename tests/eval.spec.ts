import {core as mx} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('eval', () => {
  it('eval', () => {
    const arrs: mx.array[] = [];
    for (let i = 0; i < 4; i++) {
      arrs.push(mx.ones([2, 2]));
    }
    mx.eval(...arrs);
    for (const x of arrs)
      assert.deepEqual(x.tolist(), [[1, 1], [1, 1]]);
  });

  it('retainGraph', () => {
    const fun = x => {
      let y = mx.multiply(3, x);
      mx.eval(y);
      return mx.multiply(2, y);
    };

    const dfunDx = mx.grad(fun);
    const y = dfunDx(mx.array(1.0));
    assert.equal(y.item(), 6.0);
  });

  it('evalMixed', () => {
    const one = mx.array(1);
    let x = mx.add(mx.add(one, 1), 1);
    let y = 0;
    let z = 'hello' as unknown;  // pass typecheck to test native code
    mx.eval([x, y, z]);
    assert.equal(x.item(), 3);
  });

  it('asyncEval', () => {
    const one = mx.array(1);
    let x = mx.add(mx.add(one, one), one);
    mx.asyncEval(x);
    assert.equal(x.item(), 3);

    x = mx.add(mx.add(one, one), one);
    assert.equal(x.item(), 3);

    x = mx.array([1, 2, 3]);
    let y = mx.multiply(2, x);
    mx.asyncEval(y);
    const z = mx.multiply(2, y);
    mx.asyncEval(z);
    assert(mx.arrayEqual(y, mx.array([2, 4, 6])));
    assertArrayAllTrue(mx.arrayEqual(z, mx.array([4, 8, 12])));
  });

  it('asyncEvalTwice', () => {
    for (let i = 0; i < 1000; ++i) {
      let x = mx.add(mx.add(1, 1), 1);
      mx.asyncEval(x);
      let y = mx.add(x, 1);
      mx.asyncEval(y);
      assert.equal(x.item(), 3);
      assert.equal(y.item(), 4);
    }
  });

  it('asyncEvalInTrace', () => {
    const fun = (x) => {
      let y = mx.add(x, 1.0);
      mx.asyncEval(y);
      return mx.exp(y);
    };

    assert.throws(() => {
      mx.grad(fun)(mx.array(1.0));
    }, Error);
  });

  it('asyncEvalIntoEval', () => {
    let x = mx.array(1);
    let y = mx.add(x, 1);
    mx.asyncEval(y);
    let a = mx.subtract(y, 10);
    let b = mx.abs(a);
    assert.equal(b.item(), 8);
  });

  it('asyncEvalIntoEvalDiffStream', () => {
    let s = mx.newStream(mx.cpu);
    let x = mx.array(0);
    let y = mx.subtract(x, 5);
    mx.asyncEval(y);
    let z = mx.abs(y, s);
    assert.equal(z.item(), 5);
  });

  it('evalSlowFastMultiStream', () => {
    let x = mx.ones([8000]);
    let y = mx.abs(mx.array(-1.0));
    for (let i = 0; i < 20; i++) {
      x = mx.add(x, mx.array(1.0));
    }
    let z = mx.add(x, y, mx.cpu);
    assertArrayAllTrue(mx.allclose(z, mx.full([8000], 22.0)));

    x = mx.ones([8000]);
    y = mx.abs(mx.array(-1.0));
    for (let i = 0; i < 20; i++) {
      x = mx.add(x, mx.array(1.0));
    }
    z = mx.add(y, x, mx.cpu);
    assertArrayAllTrue(mx.allclose(z, mx.full([8000], 22.0)));
  });

  it('multiOutputEvalDuringTransform', () => {
    const x = mx.random.uniform(0, 1, [1024]);
    const y = mx.ones([1024]);
    mx.eval(x, y);

    const fn = (x: mx.array) => {
      const [a, b] = mx.divmod(x, x);
      mx.eval(a);
      return a;
    };

    let out = mx.vjp(fn, [x], [y]);
    out = mx.vjp(fn, [x], [y]);
    if (mx.metal.isAvailable()) {
      const peakMem = mx.metal.getPeakMemory();
      out = mx.vjp(fn, [x], [y]);
      assert.equal(peakMem, mx.metal.getPeakMemory());
    }
  });
});
