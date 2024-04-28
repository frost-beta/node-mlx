import {core as mx} from '..';
import {assertArrayAllTrue} from './utils';
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

  it('grad', () => {
    const fun1 = (x) => mx.multiply(x, x);

    let [value, dfdx] = mx.valueAndGrad(fun1)(mx.array(0.5));
    assert.equal(value.item(), 0.25);
    assert.equal(dfdx.item(), 1.0);

    dfdx = mx.grad(fun1)(mx.array(0.5));
    assert.equal(dfdx.item(), 1.0);

    let df2dx2 = mx.grad(mx.grad(fun1))(mx.array(0.5));
    assert.equal(df2dx2.item(), 2.0);
    let df3dx3 = mx.grad(mx.grad(mx.grad(fun1)))(mx.array(0.5));
    assert.equal(df3dx3.item(), 0.0);

    const fun2 = (x, y) => mx.multiply(x, y);
    let x = mx.array(2.0);
    let y = mx.array(3.0);
    dfdx = mx.grad(fun2, 0)(x, y);
    assert.equal(dfdx.item(), 3.0);
    dfdx = mx.grad(fun2, 1)(x, y);
    assert.equal(dfdx.item(), 2.0);
  });

  it('gradTrees', () => {
    let fun = (x, y) => mx.multiply(x, y);
    let [value, dfdx] = mx.valueAndGrad(fun, [0, 1])(mx.array(0.5), mx.array(2.0));
    assert.equal(value.item(), 1.0);
    assert.equal(dfdx[0].item(), 2.0);
    assert.equal(dfdx[1].item(), 0.5);

    fun = (x, y) => mx.multiply(x, y);
    [value, dfdx] = mx.valueAndGrad(fun, 1)(mx.array(0.5), mx.array(2.0));
    assert.equal(value.item(), 1.0);
    assert.equal(dfdx.item(), 0.5);
  });

  it('captured', () => {
    const a = mx.array(5.0);
    const f = (x) => mx.add(a, x);
    const g = () => mx.add(a, a);
    const h = (x) => mx.add(x, x);

    const dfdx = mx.grad(f);
    assert.equal(dfdx(a).item(), 1.0);

    const dgdx = mx.grad(g);
    assert.equal(dgdx(a).item(), 0.0);

    const dhdx = mx.grad(h);
    assert.equal(dhdx(a).item(), 2.0);

    const d2fdx2 = mx.grad(dfdx);
    assert.equal(d2fdx2(a).item(), 0.0);

    const d2gdx2 = mx.grad(dgdx);
    assert.equal(d2gdx2(a).item(), 0.0);

    const d2hdx2 = mx.grad(dhdx);
    assert.equal(d2hdx2(a).item(), 0.0);
  });

  it('stopGradient', () => {
    const shapeIn = [4, 4];
    const wIn = mx.ones(shapeIn);
    const xIn = mx.ones(shapeIn);
    const cotan = mx.ones(shapeIn);

    const h = (w, x) => {
      let x1 = mx.multiply(2, x);
      let y = mx.stopGradient(x1);
      let y1 = mx.multiply(3, y);
      return mx.matmul(w, y1);
    };

    let [vals, vjps] = mx.vjp(h, [wIn, xIn], [cotan]);
    mx.eval(...vjps);

    assertArrayAllTrue(mx.allclose(vjps[0], mx.multiply(24.0, mx.ones(shapeIn))));
    assertArrayAllTrue(mx.allclose(vjps[1], mx.zeros(shapeIn)));

    const g = x => h(wIn, x);
    [vals, vjps] = mx.vjp(g, [xIn], [cotan]);
    mx.eval(...vjps);

    assertArrayAllTrue(mx.allclose(vjps[0], mx.zeros(shapeIn)));
  });

  it('updateState', () => {
    let y = mx.array([1.0]);
    let state = mx.zeros([2]);

    const fn = (y, x) => {
      x = mx.multiply(y, x);
      state = mx.add(state, x);
      return x.sum();
    };

    const x = mx.ones([2]);
    mx.grad(fn)(y, x);
    mx.eval(state);
    assertArrayAllTrue(mx.allclose(state, mx.ones([2])));
  });

  it('scatterVjp', () => {
    const fun1 = (x, idx) => {
      x.indexPut_(idx.astype(mx.int32), 2.0);
      return x.sum();
    };

    let dfdx = mx.grad(fun1)(mx.array([1.0, 2.0, 3.0]), mx.array([1]));

    assertArrayAllTrue(mx.arrayEqual(dfdx, mx.array([1.0, 0.0, 1.0])));
    assert.equal(dfdx.dtype, mx.float32);

    let y = mx.array([0.0, 1.0, 2.0]);

    const fun2 = (x, idx) => {
      y.indexPut_(idx.astype(mx.int32), x);
      return y.sum();
    };

    dfdx = mx.grad(fun2)(mx.array([2.0]), mx.array([1]));

    assertArrayAllTrue(mx.arrayEqual(dfdx, mx.array([1.0])));
    assert.equal(dfdx.dtype, mx.float32);
  });

  it('scatterMaxVjp', () => {
    const fun = (src, updates) => {
      let x = src.at(1).maximum(updates);
      return x;
    };

    let cotan = mx.array([4.0, 5.0, 6.0]);
    let [_, vjps] = mx.vjp(fun, [mx.array([1.0, 2.0, 3.0]), mx.array([[3.0]])], [cotan]);
    mx.eval(...vjps);

    assertArrayAllTrue(mx.allclose(vjps[0], mx.array([4.0, 0.0, 6.0])));
    assertArrayAllTrue(mx.allclose(vjps[1], mx.array([5.0])));

    cotan = mx.array([[4.0], [5.0], [6.0]]);
    [_, vjps] = mx.vjp(fun, [mx.array([[1.0], [2.0], [3.0]]), mx.array([[[2.0]]])], [cotan]);
    mx.eval(...vjps);

    assertArrayAllTrue(mx.allclose(vjps[0], mx.array([[4.0], [5.0], [6.0]])));
    assertArrayAllTrue(mx.allclose(vjps[1], mx.array([[[5.0]]])));
  });

  it('scatterMinVjp', () => {
    const fun = (src, updates) => {
      let x = src.at(1).minimum(updates);
      return x;
    };

    let cotan = mx.array([4.0, 5.0, 6.0]);
    let [, vjps] = mx.vjp(fun, [mx.array([1.0, 2.0, 3.0]), mx.array([[3.0]])], [cotan]);
    mx.eval(...vjps);

    assertArrayAllTrue(mx.allclose(vjps[0], mx.array([4.0, 5.0, 6.0])));
    assertArrayAllTrue(mx.allclose(vjps[1], mx.array([0.0])));

    cotan = mx.array([[4.0], [5.0], [6.0]]);
    [, vjps] = mx.vjp(
      fun, [mx.array([[1.0], [2.0], [3.0]]), mx.array([[[2.0]]])], [cotan]
    );
    mx.eval(...vjps);

    assertArrayAllTrue(mx.allclose(vjps[0], mx.array([[4.0], [5.0], [6.0]])));
    assertArrayAllTrue(mx.allclose(vjps[1], mx.array([[[5.0]]])));
  });

  it('splitAgainstSlice', () => {
    const fSplit = x => {
      const [a, , b] = x.split(3, -1);
      return mx.multiply(a, b).sum();
    };

    const fSlice = x => {
      const step = x.shape[x.shape.length - 1] / 3;
      const a = x.index('...', mx.Slice(null, step));
      const b = x.index('...', mx.Slice(-step));
      return mx.multiply(a, b).sum();
    };

    const x = mx.random.uniform(0, 1, [100, 300]);
    mx.eval(x);

    const df1 = mx.grad(fSplit);
    const df2 = mx.grad(fSlice);

    assertArrayAllTrue(mx.allclose(df1(x), df2(x)));
  });

  it('vjpTypes', () => {
    const fun1 = x => x;

    [mx.float16, mx.bfloat16, mx.float32].forEach(t => {
      const out = mx.grad(fun1)(mx.array(1.0, t));
      assert.equal(out.dtype, t);
    });

    const fun2 = x => x.sum();

    [mx.float16, mx.bfloat16, mx.float32].forEach(t => {
      const out = mx.grad(fun2)(mx.array(1.0, t));
      assert.equal(out.dtype, t);
    });

    const fun3 = (x, y) => mx.sum(mx.add(x, y));

    [mx.float16, mx.bfloat16, mx.float32].forEach(t => {
      const out = mx.grad(fun3)(mx.array(1.0, t), mx.array(1.0, t));
      assert.equal(out.dtype, t);
    });
  });

  it('powerGrad', () => {
    let x = mx.array(0.0);
    let g = mx.grad(x => mx.power(x, 2))(x);
    assert.equal(g.index().item(), 0.0);

    x = mx.array(0.0);
    g = mx.grad(x => mx.power(x, 1.5))(x);
    assert.equal(g.index().item(), 0.0);

    x = mx.array(2.0);
    g = mx.grad(x => mx.power(x, 2))(x);
    assert.equal(g.index().item(), 4.0);
  });

  it('evalInGrad', () => {
    const arr = mx.array([1.0]);
    const cotan = mx.array([1.0, 1.0]);
    let y = mx.array([2.0, 2.0]);

    const func1 = x => {
      x = mx.add(x, y);
      const cond = mx.less(x, 1);
      cond.tolist();
      return mx.power(x, 2);
    };

    let [, vjps] = mx.vjp(func1, [arr], [cotan]);
    assert.equal(vjps[0].index().item(), 12.0);

    const func2 = x => {
      x = mx.add(x, mx.array([1.0, 1.0]));
      mx.eval(x);
      return mx.power(x, 2);
    };

    [, vjps] = mx.vjp(func2, [arr], [cotan]);
    assert.equal(vjps[0].index().item(), 8.0);
  });

  it('subtractPowerGrad', () => {
    const fun = (x, y) => {
      const res = mx.subtract(x, y);
      return mx.power(res, x);
    };

    const grad = mx.grad(fun)(mx.array(1.0), mx.array(1.0));
    assert.equal(grad.index().item(), 1.0);
  });
});
