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

    const fun3 = (x, y): mx.array => x;
    [value, dfdx] = mx.valueAndGrad(fun3)(mx.array(2.0), 'hello');
    assert.equal(value.item(), 2.0);
    assert.equal(dfdx.item(), 1.0);

    dfdx = mx.grad(fun3)(mx.array(2.0), 'hello');
    assert.equal(dfdx.item(), 1.0);

    const fun4 = x => 'hello';
    assert.throws(() => {
      mx.grad(fun4)(mx.array(2.0));
    }, Error);

    const fun5 = x => x;
    assert.throws(() => {
      mx.grad(fun5, 2)(mx.array(2.0));
    }, Error);
    assert.throws(() => {
      mx.grad(fun5, -2)(mx.array(2.0));
    }, Error);
    assert.throws(() => {
      mx.grad(fun5)('hello');
    }, Error);

    const fun6 = x => mx.sum(x, true);
    assert.throws(() => {
      mx.grad(fun6)(mx.ones([2, 2]));
    }, Error);
  });

  it('gradTrees', () => {
    let fun = (x, y) => mx.multiply(x, y);
    let [value, dfdx] = mx.valueAndGrad(fun, [0, 1])(mx.array(0.5), mx.array(2.0));
    assert.equal(value.item(), 1.0);
    assert.isTrue(dfdx instanceof Array);
    assert.equal(dfdx[0].item(), 2.0);
    assert.equal(dfdx[1].item(), 0.5);

    fun = (x, y) => mx.multiply(x, y);
    [value, dfdx] = mx.valueAndGrad(fun, 1)(mx.array(0.5), mx.array(2.0));
    assert.equal(value.item(), 1.0);
    assert.equal(dfdx.item(), 0.5);

    let fun2 = p => mx.multiply(p['x'], p['y']);
    [value, dfdx] = mx.valueAndGrad(fun2)({x: mx.array(0.5), y: mx.array(2.0)});
    assert.equal(value.item(), 1.0);
    assert.equal(dfdx['x'].item(), 2.0);
    assert.equal(dfdx['y'].item(), 0.5);

    fun2 = p => mx.multiply(p['x'], p['y']);
    assert.throws(() => {
      mx.valueAndGrad(fun2)({x: 'string', y: mx.array(2.0)});
    }, Error);
    assert.throws(() => {
      mx.valueAndGrad(fun2, [0, 1])({x: mx.array(0.5), y: mx.array(2.0)});
    }, Error);

    fun = (p, b) => mx.multiply(mx.square(p[0]['foo'][2]), b);
    [value, dfdx] = mx.valueAndGrad(fun)([{ 'foo': [[], [], mx.array(2.0)] }], mx.array(0.5));
    assert.equal(value.item(), 2.0);
    assert.equal(dfdx[0]['foo'][2].item(), 2.0);

    fun = (x) => x;
    assert.throws(() => {
      mx.valueAndGrad(fun, [0, 0]);
    }, Error);
  });


  it('auxiliaryValues', () => {
    const fun = (x, y) => {
      let l = mx.multiply(x, y).sum();
      let extra = {
        'loss': l,
        'foo': mx.add(y.square(), x.square()),
        'bar': [1, 2, 3, y, x]
      };
      return [l, extra];
    };

    const funValueGrad = mx.valueAndGrad(fun);
    const funGrad = mx.grad(fun);

    const [[loss, a], b] = funValueGrad(mx.ones([2, 2]), mx.ones([2, 2])) as any;

    assert.equal(a['loss'].item(), 4);
    assertArrayAllTrue(mx.arrayEqual(b, mx.ones([2, 2])));
    assertArrayAllTrue(mx.arrayEqual(a['foo'], mx.multiply(2, mx.ones([2, 2]))));
    assert.deepEqual(a['bar'].slice(0, 3), [1, 2, 3]);
    assertArrayAllTrue(mx.arrayEqual(a['bar'][3], mx.ones([2, 2])));
    assertArrayAllTrue(mx.arrayEqual(a['bar'][4], mx.ones([2, 2])));

    assert.throws(() => {
      funGrad(mx.ones([2, 2]), mx.ones([2, 2]));
    }, Error);
  });

  it('captured', () => {
    const a = mx.array(5.0);
    const f = (x) => mx.add(a, x);
    const g = () => mx.add(a, a);
    const h = (x) => mx.add(x, x);

    const dfdx = mx.grad(f);
    assert.equal(dfdx(a).item(), 1.0);

    const dgdx = mx.grad(g) as (x: mx.array) => mx.array;
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
      return mx.sum(x);
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
    const fun1 = (x: mx.array) => x;

    [mx.float16, mx.bfloat16, mx.float32].forEach(t => {
      const out = mx.grad(fun1)(mx.array(1.0, t));
      assert.equal(out.dtype, t);
    });

    const fun2 = (x: mx.array) => x.sum();

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
    let g = mx.grad((x: mx.array) => mx.power(x, 2))(x);
    assert.equal(g.index().item(), 0.0);

    x = mx.array(0.0);
    g = mx.grad((x: mx.array) => mx.power(x, 1.5))(x);
    assert.equal(g.index().item(), 0.0);

    x = mx.array(2.0);
    g = mx.grad((x: mx.array) => mx.power(x, 2))(x);
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

  it('cumprodGrad', () => {
    const funA = (y: mx.array) => {
      return mx.cumprod(y).sum();
    };

    let y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0]);
    let out = mx.grad(funA)(y);
    let expected = mx.array([20.0, 38.0, 18.0, 16.0, 8.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0]);
    out =  mx.grad(funA)(y);
    expected = mx.array([1.0, 38.0, 0.0, 0.0, 0.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0]);
    out = mx.grad(funA)(y);
    expected = mx.array([1.0, 6.0, 0.0, 0.0, 0.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    const funB = (y: mx.array) => {
      return mx.cumprod(y, undefined, undefined, false).sum();
    };

    y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0]);
    out = mx.grad(funB)(y);
    expected = mx.array([8.0, 14.0, 6.0, 4.0, 0.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0]);
    out = mx.grad(funB)(y);
    expected = mx.array([1.0, 14.0, 0.0, 0.0, 0.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0]);
    out = mx.grad(funB)(y);
    expected = mx.array([1.0, 6.0, 0.0, 0.0, 0.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    const funC = (y: mx.array) => {
      return mx.cumprod(y, undefined, true, false).sum();
    };

    y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0]);
    out = mx.grad(funC)(y);
    expected = mx.array([0.0, 12.0, 12.0, 15.0, 11.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0]);
    out = mx.grad(funC)(y);
    expected = mx.array([0.0, 12.0, 6.0, 9.0, 7.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0]);
    out = mx.grad(funC)(y);
    expected = mx.array([0.0, 0.0, 0.0, 9.0, 1.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    const funD = (y: mx.array) => {
      return mx.cumprod(y, undefined, true).sum();
    };

    y = mx.array([2.0, 1.0, 2.0, 2.0, 3.0]);
    out = mx.grad(funD)(y);
    expected = mx.array([12.0, 36.0, 24.0, 27.0, 19.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 2.0, 3.0]);
    out = mx.grad(funD)(y);
    expected = mx.array([0.0, 36.0, 6.0, 9.0, 7.0]);
    assertArrayAllTrue(mx.allclose(out, expected));

    y = mx.array([2.0, 0.0, 2.0, 0.0, 3.0]);
    out = mx.grad(funD)(y);
    expected = mx.array([0.0, 0.0, 0.0, 9.0, 1.0]);
    assertArrayAllTrue(mx.allclose(out, expected));
  });

  // FIXME(zcbenz): Figure out why this test is failing.
  it.skip('topkGrad', () => {
    const a = mx.array([[1, 2, 6, 4, 5], [9, 5, 6, 7, 8]], mx.float32);

    const fun = (x: mx.array) => {
      return mx.topk(x, 2);
    }

    const out = mx.vjp(fun, [a], [mx.ones([2, 2])])[1][0];
    const expected = mx.array([[0, 0, 1, 0, 1], [1, 0, 0, 0, 1]], mx.float32);
    assertArrayAllTrue(mx.arrayEqual(out, expected));
  });

  it('flattenUnflattenVjps', () => {
    const fun1 = (x: mx.array) => {
      const y = mx.unflatten(x, 0, [2, 2]);
      return y.sum();
    }

    let x = mx.zeros([4, 8]);
    assert.deepEqual(mx.grad(fun1)(x).shape, [4, 8]);

    const fun2 = (x: mx.array) => {
      const y = mx.flatten(x, 0, 2);
      return y.sum();
    }

    x = mx.zeros([2, 4, 8]);
    assert.deepEqual(mx.grad(fun2)(x).shape, [2, 4, 8]);
  });

  it('concatenateVjps', () => {
    const fun = (x, y) => {
      return mx.concatenate([x, y]);
    }

    const x = mx.array([1, 2, 3], mx.float32);
    const y = mx.array([1, 2, 3], mx.float16);
    const grads = mx.vjp(fun, [x, y], [mx.ones([6])])[1];
    assertArrayAllTrue(mx.allclose(grads[0], mx.ones([3])));
    assertArrayAllTrue(mx.allclose(grads[1], mx.ones([3])));
    assert.equal(grads[0].dtype, mx.float32);
    assert.equal(grads[1].dtype, mx.float16);
  });

  it('matmulJvps', () => {
    let a = mx.random.uniform(0, 1, [4, 4]);
    let b = mx.random.uniform(0, 1, [4, 4]);
    let c = mx.random.uniform(0, 1, [4, 4]);
    let d = mx.random.uniform(0, 1, [4, 4]);

    let _: mx.array;
    let tangents: mx.array[];
    [, tangents] = mx.jvp((a) => mx.matmul(a, b), [a], [c]);
    assertArrayAllTrue(mx.allclose(tangents[0], mx.matmul(c, b)));

    [, tangents] = mx.jvp((b) => mx.matmul(a, b), [b], [d]);
    assertArrayAllTrue(mx.allclose(tangents[0], mx.matmul(a, d)));

    [, tangents] = mx.jvp((a, b) => mx.matmul(a, b), [a, b], [c, d]);
    assertArrayAllTrue(mx.allclose(tangents[0], mx.add(mx.matmul(a, d), mx.matmul(c, b))));

    let x = mx.random.uniform(0, 1, [4, 4]);
    let y = mx.random.uniform(0, 1, [4, 4]);
    let z = mx.random.uniform(0, 1, [4, 4]);

    let tangent: mx.array;
    let expected: mx.array;
    [, [tangent]] = mx.jvp((a, b, c) => mx.add(mx.matmul(a, b), c), [a, b, c], [x, y, z]);
    [, [expected]] = mx.jvp((a, b, c) => mx.addmm(c, a, b), [a, b, c], [x, y, z]);
    assertArrayAllTrue(mx.allclose(tangent, expected));

    [, [tangent]] = mx.jvp((a, c) => mx.add(mx.matmul(a, b), c), [a, c], [x, z]);
    [, [expected]] = mx.jvp((a, c) => mx.addmm(c, a, b), [a, c], [x, z]);
    assertArrayAllTrue(mx.allclose(tangent, expected));

    [, [tangent]] = mx.jvp((b, c) => mx.add(mx.matmul(a, b), c), [b, c], [y, z]);
    [, [expected]] = mx.jvp((b, c) => mx.addmm(c, a, b), [b, c], [y, z]);
    assertArrayAllTrue(mx.allclose(tangent, expected));

    [, [tangent]] = mx.jvp((c) => mx.add(mx.matmul(a, b), c), [c], [z]);
    [, [expected]] = mx.jvp((c) => mx.addmm(c, a, b), [c], [z]);
    assertArrayAllTrue(mx.allclose(tangent, expected));
  });

  it('putAlongAxisGrads', () => {
    const a = mx.zeros([5, 1]);
    const b = mx.ones([2, 1]);

    const fun = (a: any, b: any) => {
      const idx = mx.array([[0], [3]], mx.int32);
      return mx.putAlongAxis(a, idx, b, 0);
    };

    // Test VJP
    const cotan = mx.full([5, 1], 2.0);
    const [, [da, db]] = mx.vjp(fun, [a, b], [cotan]);
    const expectedDa = mx.array([0.0, 2.0, 2.0, 0.0, 2.0]).index(mx.Slice(), null);
    const expectedDb = mx.array([2.0, 2.0]).index(mx.Slice(), null);
    assertArrayAllTrue(mx.allclose(expectedDa, da));
    assertArrayAllTrue(mx.allclose(expectedDb, db));

    // Test JVP
    const tanA = mx.full([5, 1], 2.0);
    const tanB = mx.full([2, 1], 3.0);
    const [, [jout]] = mx.jvp(fun, [a, b], [tanA, tanB]);
    const expected = mx.array([3.0, 2.0, 2.0, 3.0, 2.0]).index(mx.Slice(), null);
    assertArrayAllTrue(mx.allclose(expected, jout));

    const funSingle = (a: mx.array) => {
      const idx = mx.array([[0], [3]], mx.int32);
      return mx.putAlongAxis(a, idx, b, 0);
    };

    const [, [joutSingle]] = mx.jvp(funSingle, [a], [tanA]);
    const expectedSingle = mx.array([0.0, 2.0, 2.0, 0.0, 2.0]).index(mx.Slice(), null);
    assertArrayAllTrue(mx.allclose(expectedSingle, joutSingle));
  });

  it('sliceGrads', () => {
    // Slice.
    const fun = (a: mx.array) => a.index(mx.Slice(5, -6, -1));

    let a = mx.ones([5]);
    let cotan = mx.random.uniform(0, 1, [5]);
    let [, [grad]] = mx.vjp(fun, [a], [cotan]);
    assert.deepEqual(grad.tolist(),
                     cotan.index(mx.Slice(null, null, -1)).tolist());

    let tan = mx.random.uniform(0, 1, [5]);
    mx.eval(tan);
    [, [grad]] = mx.jvp(fun, [a], [tan]);
    assert.deepEqual(grad.tolist(),
                     tan.index(mx.Slice(null, null, -1)).tolist());

    // Slice update.
    const fun2 = (a: mx.array, b: mx.array) => {
      a.indexPut_(mx.Slice(4, -5, -2), b);
      return a;
    };

    a = mx.ones([4]);
    const b = mx.zeros([2]);

    cotan = mx.random.uniform(0, 1, [4]);
    let [, [gradA, gradB]] = mx.vjp(fun2, [a, b], [cotan]);
    const expectedA = mx.array(cotan);
    expectedA.indexPut_(mx.Slice(1, null, 2), 0);
    assert.deepEqual(gradA.tolist(), expectedA.tolist());
    assert.deepEqual(gradB.tolist(), cotan.index(mx.Slice(4, -5, -2)).tolist());

    const tanA = mx.random.uniform(0, 1, [4]);
    const tanB = mx.random.uniform(0, 1, [2]);
    [, [grad]] = mx.jvp(fun2, [a, b], [tanA, tanB]);
    let expected = mx.array(tanA);
    expected.indexPut_(mx.Slice(4, -5, -2), tanB);
    assertArrayAllTrue(mx.allclose(grad, expected));
  });

  // FIXME(zcbenz): https://github.com/ml-explore/mlx/pull/1961
  xit('gradWithInplaceUpdate', () => {
    const lossFn = (model: mx.array[]) => {
      model[1] = mx.array(2.0);
      return model[0];
    }

    const model = [
      mx.array(0.0),
      mx.array(1.0),
    ];

    const gradFn = mx.grad(lossFn);
    gradFn(model);
    assert.equal(model[1].item(), 2.0);
  });
});
