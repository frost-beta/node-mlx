import {core as mx} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('compile', function() {
  beforeEach(function() {
    // FIXME(zcbenz): Compilation fails on QEMU in CI.
    if (process.env.CI == 'true' &&
        process.platform == 'linux' &&
        process.arch == 'arm64') {
      this.skip();
    }
  });

  this.timeout(10 * 1000);

  it('simpleCompile', () => {
    const fun = (x, y) => mx.add(x, y);

    let compiledFn = mx.compile(fun);
    let x = mx.array(1.0);
    let y = mx.array(1.0);
    let out = compiledFn(x, y);
    assert.equal(out.item() as number, 2.0);

    out = compiledFn(x, y);
    assert.equal(out.item() as number, 2.0);

    x = mx.array([1.0, 2.0]);
    out = compiledFn(x, y);
    assertArrayAllTrue(mx.arrayEqual(out, mx.array([2.0, 3.0])));

    y = mx.array([1.0, 2.0]);
    out = compiledFn(x, y);
    assertArrayAllTrue(mx.arrayEqual(out, mx.array([2.0, 4.0])));

    x = mx.array([1, 2], mx.int32);
    y = mx.array([1, 2], mx.int32);
    out = compiledFn(x, y);
    assert.equal(out.dtype, mx.int32);
    assertArrayAllTrue(mx.arrayEqual(out, mx.array([2, 4])));
  });

  it('compileGrad', () => {
    const lossFn = x => mx.exp(x).sum();
    const gradFn = mx.grad(lossFn);

    let x = mx.array([0.5, -0.5, 1.2]);
    let dfdx = gradFn(x);
    let compileGradFn = mx.compile(gradFn);
    let cDfdx = compileGradFn(x);

    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));

    cDfdx = compileGradFn(x);
    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));

    cDfdx = mx.compile(gradFn)(x);
    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));

    let loss, cLoss;
    const valAndGradFn = mx.valueAndGrad(lossFn);
    [loss, dfdx] = valAndGradFn(x);
    [cLoss, cDfdx] = mx.compile(valAndGradFn)(x);

    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));
    assertArrayAllTrue(mx.equal(cLoss, loss));
  });


  it('compileInputsWithPrimitives', () => {
    let x = mx.array([1, 2, 3]);
    let y = mx.array([1, 2, 3]);
    for (let i = 0; i < 5; i++) {
      x = mx.add(x, y);
      y = mx.add(y, 1);
    }

    const fun = (x, y) => {
      return mx.multiply(x, y);
    };
    const out = fun(x, y);

    x = mx.array([1, 2, 3]);
    y = mx.array([1, 2, 3]);
    for (let i = 0; i < 5; i++) {
      x = mx.add(x, y);
      y = mx.add(y, 1);
    }

    const cOut = mx.compile(fun)(x, y);
    assertArrayAllTrue(mx.arrayEqual(out, cOut));

    const cOutRecovered = mx.compile(fun)(x, y);
    assertArrayAllTrue(mx.arrayEqual(out, cOutRecovered));
  });

  it('compileWithClosure', () => {
    let x = mx.array(1);

    const closure = (y) => {
      return mx.add(x, y);
    }

    let compiled = mx.compile(closure);
    let out = compiled(mx.array(1));
    assert.equal(out.item(), 2);

    // Try again
    out = compiled(mx.array(1));
    assert.equal(out.item(), 2);

    // Change the shape of the enclosed variable
    x = mx.array([1, 2]);
    out = compiled(mx.array(1));

    // We still get the original input (closures are not updated)
    assert.equal(out.item(), 2);

    // Try with a tree of enclosed variables
    let x2 = {a: mx.array(1), b: mx.array(2)};

    const closure2 = (y) => {
      return mx.add(mx.add(x2['a'], y), x2['b']);
    }

    compiled = mx.compile(closure2);
    out = compiled(mx.array(1));
    assert.equal(out.item(), 4);

    // Change the shape of one input
    x2['a'] = mx.array([4, 5]);
    out = compiled(mx.array(1));
    assert.equal(out.item(), 4);

    x2['b'] = mx.array([-6, -8]);
    out = compiled(mx.array(1));
    assert.equal(out.item(), 4);

    // Enclosed variable is not evaluated yet
    x = mx.array(1);
    x = mx.add(x, x);

    const closure3 = (y) => {
      return mx.add(x, y);
    }

    compiled = mx.compile(closure3);
    out = compiled(mx.array(2));
    assert.equal(out.item(), 4);

    // And again
    out = compiled(mx.array(2));
    assert.equal(out.item(), 4);
  });

  it('functionCreatesArray', () => {
    const fun = x => mx.add(x, mx.array(1));
    const cfun = mx.compile(fun);

    let out = cfun(mx.array(3));
    assert.equal(out.item(), 4);

    out = cfun(mx.array(3));
    assert.equal(out.item(), 4);
  });

  // TODO(zcbenz): Add test_enable_disable after implementing export_to_dot.

  it('compileTwoInputGrad', () => {
    const loss = (w, x) => {
      let y = mx.multiply(x, w);
      return mx.multiply(y, mx.exp(y)).sum();
    };

    let x = mx.array([1.0, 0.5, 2.0, -0.5]);
    let w = mx.array([-1.0, 0.3, 1.0, -0.9]);

    const expectedGrad = mx.grad(loss)(w, x);
    const compiledGrad = mx.compile(mx.grad(loss))(w, x);
    assertArrayAllTrue(mx.allclose(expectedGrad, compiledGrad));
  });

  it('vmapCompiled', () => {
    const simpleUnary = x => mx.negative(mx.exp(x));

    let x = mx.array([[1.0, 2.0], [2.0, 3.0]]);

    let expectedOut = mx.vmap(simpleUnary)(x);
    let out = mx.vmap(mx.compile(simpleUnary))(x);
    assertArrayAllTrue(mx.allclose(expectedOut, out));

    const simpleBinary = (x, y) => mx.abs(mx.add(mx.exp(mx.add(x, y)), y));

    x = mx.array([[1.0, -3.0], [0.5, -0.5]]);
    let y = mx.array([[2.0, -1.0], [0.25, -0.25]]);

    expectedOut = mx.vmap(simpleBinary)(x, y);
    out = mx.vmap(mx.compile(simpleBinary))(x, y);
    assertArrayAllTrue(mx.allclose(expectedOut, out));

    expectedOut = mx.vmap(simpleBinary, [0, 1])(x, y);
    out = mx.vmap(mx.compile(simpleBinary), [0, 1])(x, y);
    assertArrayAllTrue(mx.allclose(expectedOut, out));

    y = mx.array([0.25, -0.25]);
    expectedOut = mx.vmap(simpleBinary, [0, -1])(x, y);
    out = mx.vmap(mx.compile(simpleBinary), [0, -1])(x, y);
    assertArrayAllTrue(mx.allclose(expectedOut, out));

    const simpleUnaryOuter = x => {
      x = mx.abs(x);
      const simpleUnaryInner = mx.compile((z: mx.array) => mx.negative(mx.exp(z)));
      return simpleUnaryInner(x);
    };

    expectedOut = mx.negative(mx.exp(mx.abs(x)));
    out = mx.vmap(simpleUnaryOuter)(x);
    assertArrayAllTrue(mx.allclose(expectedOut, out));
  });

  it('vjpVjpCompiled', () => {
    const simpleUnary = x => mx.negative(mx.exp(x));

    let x = mx.array([[1.0, 2.0], [2.0, 3.0]]);
    let y = mx.array([[1.0, 1.0], [1.0, 1.0]]);

    let [expectedOut, expectedVjpOut] = mx.vjp(simpleUnary, [x], [y]);
    let [out, vjpOut] = mx.vjp(mx.compile(simpleUnary), [x], [y]);
    assertArrayAllTrue(mx.allclose(expectedVjpOut[0], vjpOut[0]));
    assertArrayAllTrue(mx.allclose(expectedOut[0], out[0]));

    let jvpOut, expectedJvpOut;
    [expectedOut, expectedJvpOut] = mx.jvp(simpleUnary, [x], [y]);
    [out, jvpOut] = mx.jvp(mx.compile(simpleUnary), [x], [y]);
    assertArrayAllTrue(mx.allclose(expectedJvpOut[0], jvpOut[0]));
    assertArrayAllTrue(mx.allclose(expectedOut[0], out[0]));

    const simpleBinary = (x, y) => mx.abs(mx.add(mx.exp(mx.add(x, y)), y));

    x = mx.array([[1.0, -3.0], [0.5, -0.5]]);
    y = mx.array([[2.0, -1.0], [0.25, -0.25]]);
    const cotans = mx.onesLike(x);

    [expectedOut, expectedVjpOut] = mx.vjp(simpleBinary, [x, y], [cotans]);
    [out, vjpOut] = mx.vjp(mx.compile(simpleBinary), [x, y], [cotans]);
    assertArrayAllTrue(mx.allclose(expectedOut[0], out[0]));
    assertArrayAllTrue(mx.allclose(expectedVjpOut[0], vjpOut[0]));
    assertArrayAllTrue(mx.allclose(expectedVjpOut[1], vjpOut[1]));

    const tans = [mx.onesLike(x), mx.onesLike(y)];
    [expectedOut, expectedJvpOut] = mx.jvp(simpleBinary, [x, y], tans);
    [out, jvpOut] = mx.jvp(mx.compile(simpleBinary), [x, y], tans);
    assertArrayAllTrue(mx.allclose(expectedJvpOut[0], jvpOut[0]));
    assertArrayAllTrue(mx.allclose(expectedOut[0], out[0]));
  });

  it('transformOverEvalCompiled', () => {
    const outer = x => {
      let y = mx.exp(mx.abs(x));
      mx.eval(y);
      return y.sum();
    };

    let x = mx.array([2.0, -1.0, 0.5]);
    const dfdx = mx.grad(outer)(x);

    const simpleUnary = mx.compile((x: mx.array) => mx.exp(mx.abs(x)));

    const outerCompiled = x => {
      let y = simpleUnary(x);
      mx.eval(y);
      return y.sum();
    };

    const cdfdx = mx.grad(outerCompiled)(x);
    assertArrayAllTrue(mx.allclose(dfdx, cdfdx));
  });

  // TODO(zcbenz): Add test_compile_capture/test_compile_rng after implementing
  // tree flatten.

  it('shapelessCompile', () => {
    let y = 1;
    const fun = mx.compile((x: mx.array) => mx.add(x, y), true);

    let x = mx.array([1, 2]);
    assertArrayAllTrue(mx.arrayEqual(fun(x), mx.array([2, 3])));

    y = 2;
    x = mx.array([1, 2, 3]);
    assertArrayAllTrue(mx.arrayEqual(fun(x), mx.array([2, 3, 4])));

    x = mx.array([1, 2, 3], mx.int32);
    assertArrayAllTrue(mx.allclose(fun(x), mx.array([3, 4, 5])));

    x = mx.array([[1, 2, 3]]);
    assertArrayAllTrue(mx.allclose(fun(x), mx.array([[3, 4, 5]])));
  });

  it('shapelessCompileWithBroadcasts', () => {
    let x = mx.ones([2, 2]);
    let y = mx.array([2, 2]);

    const fun = (x, y) => {
      return mx.multiply(x, y);
    };

    const cfun = mx.compile(fun, true);
    assertArrayAllTrue(mx.arrayEqual(cfun(x, y), fun(x, y)));
    assertArrayAllTrue(mx.arrayEqual(cfun(y, x), fun(y, x)));
    y = mx.array([[3]]);
    assertArrayAllTrue(mx.arrayEqual(cfun(x, y), fun(x, y)));
    assertArrayAllTrue(mx.arrayEqual(cfun(y, x), fun(y, x)));
  });

  it('shapelessCompileWithReduction', function() {
    this.timeout(10 * 1000);  // slow in QEMU

    let z = 1;
    const fun = mx.compile((x: mx.array, y: mx.array) => {
      return mx.add(mx.add(x, y.sum(0, true)), z);
    }, true);

    let x = mx.ones([2, 2], mx.int32);
    let y = mx.ones([2, 2], mx.int32);
    assertArrayAllTrue(mx.arrayEqual(fun(x, y), mx.full([2, 2], 4)));
    x = mx.ones([3, 3], mx.int32);
    y = mx.ones([3, 3], mx.int32);
    z = 2;
    assertArrayAllTrue(mx.arrayEqual(fun(x, y), mx.full([3, 3], 5)));

    const x1 = mx.array([[1, 2], [3, 4], [5, 6]]);
    const x2 = mx.array([[1, 2]]);

    const fun1 = x => {
      return mx.multiply(x, x.sum(-1, true));
    };

    const cfun = mx.compile(fun1, true);
    mx.eval(cfun(x1));
    assertArrayAllTrue(mx.arrayEqual(fun1(x2), cfun(x2)));
  });

  // TODO(zcbenz): Add test_compile_with_constant after implementing tree
  // flatten.

  it('compileInf', () => {
    const fun = mx.compile((x: mx.array) => mx.isinf(mx.add(x, 2)));
    const out = fun(mx.array([0.0]));
    assert.equal(out.item(), false);
  });

  it('compileCreateList', () => {
    const fun = mx.compile(() => {
      return [mx.multiply(0.1, mx.zeros([2])), mx.multiply(0.1, mx.zeros([2]))];
    });

    const out = fun();
    mx.eval(...out);
  });

  it('compileVjp', function() {
    this.timeout(10 * 1000);

    const fun = w => {
      let w1 = mx.add(w, w);
      let w2 = mx.add(w, w);
      return mx.add(mx.matmul(w, w1), mx.matmul(w2, w2));
    }

    const step = w => {
      let [out, grad] = mx.vjp(fun, [w], [mx.array([[1.0, 1.0], [1.0, 1.0]])]);
      return [out[0], grad[0]];
    }

    let w = mx.zeros([2, 2]);
    mx.eval(w);

    let expected = step(w);
    let out = mx.compile(step)(w);
    assertArrayAllTrue(mx.allclose(expected[0], out[0]));
    assertArrayAllTrue(mx.allclose(expected[1], out[1]));

    const fun2 = (w1, w2, x) => {
      x = mx.matmul(x, w1);
      let y = mx.matmul(x, w2);
      x = mx.add(x, mx.multiply(y, y));
      return (mx.multiply(x, x)).sum();
    }

    let w1 = mx.zeros([4, 4]);
    let w2 = mx.zeros([4, 4]);
    let x = mx.zeros([4, 4]);

    const step2 = (w1: mx.array, w2: mx.array, x: mx.array) => {
      let [loss, gradient] = mx.valueAndGrad(fun2)(w1, w2, x);
      w1 = mx.add(w1, gradient);
      return [loss, w1];
    }

    mx.eval(x, w1, w2);
    expected = step2(w1, w2, x);
    out = mx.compile(step2)(w1, w2, x);

    assertArrayAllTrue(mx.allclose(expected[0], out[0]));
    assertArrayAllTrue(mx.allclose(expected[1], out[1]));
  });

  it('shapelessMean', () => {
    const mean = x => mx.mean(x, true);
    const cmean = mx.compile(mean, true);

    let x = mx.ones(2);
    let out = cmean(x);
    assertArrayAllTrue(mx.allclose(out, mean(x)));

    x = mx.ones(4);
    out = cmean(x);
    assertArrayAllTrue(mx.allclose(out, mean(x)));

    x = mx.ones(7);
    out = cmean(x);
    assertArrayAllTrue(mx.allclose(out, mean(x)));
  });

  it('compileBroadcastOnly', () => {
    const fn = a => {
      a = mx.broadcastTo(a, [1]);
      return mx.add(a, a);
    };

    const out = mx.compile(fn)(mx.array(2.0));
    assert.deepEqual(out.tolist(), [4.0]);
  });

  it('compileWithLongName', () => {
    const fn = (a, b) => {
      for (let i = 0; i < 10; i++) {
        a = mx.subtract(a, 1.0);
        b = mx.subtract(b, 1.0);
      }
      return mx.add(a, b);
    };

    const out = mx.compile(fn)(mx.array(10.0), mx.array(20.0));
    assert.equal(out.item(), 10.0);
  });

  // TODO(zcbenz): Add test_compile_multi_output after implementing tree
  // flatten.
});
