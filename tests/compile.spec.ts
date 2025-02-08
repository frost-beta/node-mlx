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
    const lossFn = (x) => {
      return mx.exp(x).sum();
    }

    const gradFn = mx.grad(lossFn);

    let x = mx.array([0.5, -0.5, 1.2]);
    let dfdx = gradFn(x);
    let compileGradFn = mx.compile(gradFn);
    let cDfdx = gradFn(x);

    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));

    // Run it again without calling compile
    cDfdx = compileGradFn(x);
    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));

    // Run it again with calling compile
    cDfdx = mx.compile(gradFn)(x);
    assertArrayAllTrue(mx.allclose(cDfdx, dfdx));

    // Value and grad
    const lossFnValGrad = (x) => {
      return [mx.exp(x).sum(), mx.sin(x)];
    }

    const valAndGradFn = mx.valueAndGrad(lossFnValGrad);
    const [[loss, val], dfdx2] = valAndGradFn(x);
    const [[cLoss, cVal], cDfdx2] = mx.compile(valAndGradFn)(x);

    assertArrayAllTrue(mx.allclose(cDfdx2, dfdx2));
    assertArrayAllTrue(mx.allclose(cLoss, loss));
    assertArrayAllTrue(mx.allclose(cVal, val));
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
  // capturing.

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

  it('shapelessCompileWithReduction', () => {
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

    const fun1 = (x: mx.array) => {
      return mx.multiply(x, x.sum(-1, true));
    };

    const cfun = mx.compile(fun1, true);
    mx.eval(cfun(x1));
    assertArrayAllTrue(mx.arrayEqual(fun1(x2), cfun(x2)));

    const fun2 = (x: mx.array) => {
      return mx.multiply(x, x.sum(-1, false));
    };
    const cfun2 = mx.compile(fun2, true);
    assertArrayAllTrue(mx.arrayEqual(fun2(x2), cfun2(x2)));
  });

  describe('shapelessCompileUnflatten', () => {
    const x = mx.zeros([1, 1, 4 * 32]);
    const fun = (x: mx.array) =>  mx.unflatten(x, -1, [4, -1]);
    assert.deepEqual(mx.compile(fun, true)(x).shape, [1, 1, 4, 32]);
  });

  describe('shapelessCompileGather', () => {
    const x = mx.zeros([1, 1, 32]);
    const fun = (x: mx.array) =>  x.index(mx.Slice(), -1, mx.Slice());
    assert.deepEqual(mx.compile(fun, true)(x).shape, [1, 32]);
  });

  describe('compileWithConstant', () => {
    it('float', () => {
      const fun = (x, y) => {
        return mx.add(x, y);
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1.0), 1.0);
      assert.equal(z.item(), 2.0);

      z = compiledFun(mx.array(1.0), 2.0);
      assert.equal(z.item(), 3.0);

      z = compiledFun(mx.array(1.0), 1.0);
      assert.equal(z.item(), 2.0);

      z = compiledFun(mx.array(1.0), 3.0);
      assert.equal(z.item(), 4.0);
    });

    it('tuple', () => {
      const fun = (x, y = [1, 2]) => {
        return mx.add(mx.add(x, y[0]), y[1]);
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1));
      assert.equal(z.item(), 4);

      z = compiledFun(mx.array(1), [2, 2]);
      assert.equal(z.item(), 5);

      z = compiledFun(mx.array(1), [2, 1]);
      assert.equal(z.item(), 4);
    });

    it('bool', () => {
      const fun = (x, y) => {
        if (y) {
          return mx.add(x, 1);
        } else {
          return mx.add(x, 2);
        }
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1), true);
      assert.equal(z.item(), 2);

      z = compiledFun(mx.array(1), false);
      assert.equal(z.item(), 3);
    });

    it('string', () => {
      const fun = (x, y) => {
        if (y === 'one') {
          return mx.add(x, 1);
        } else {
          return mx.add(x, 2);
        }
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1), 'one');
      assert.equal(z.item(), 2);

      z = compiledFun(mx.array(1), 'two');
      assert.equal(z.item(), 3);
    });

    it('nested', () => {
      const fun = (x, y) => {
        if (y[0][0] === 1) {
          return mx.add(x, 1);
        } else {
          return mx.add(x, 2);
        }
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1), [[1]]);
      assert.equal(z.item(), 2);

      z = compiledFun(mx.array(1), [[0]]);
      assert.equal(z.item(), 3);
    });

    it('loop', () => {
      const fun = (x, a, b) => {
        for (const ai of a) {
          for (const bi of b) {
            x = mx.add(mx.multiply(bi, x), ai);
          }
        }
        return x;
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1), [1, 1], [2]);
      assert.equal(z.item(), 7);

      z = compiledFun(mx.array(1), [1], [1, 2]);
      assert.equal(z.item(), 5);
    });

    it('counter', () => {
      let counter = 0;
      const fun = (x, y) => {
        counter += 1;
        return mx.add(x, y);
      }

      const compiledFun = mx.compile(fun);
      let z = compiledFun(mx.array(1), 1);
      assert.equal(z.item(), 2);

      z = compiledFun(1, mx.array(1));
      assert.equal(z.item(), 2);

      assert.equal(counter, 2);
    });
  });

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

  it('compileVjp', () => {
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

    const cfun = mx.compile(mean);
    let out = cfun(mx.ones([5, 5]));
    assertArrayAllTrue(mx.allclose(out, mx.array(1)));

    const cmean = mx.compile(mean, true);

    let x = mx.ones(2);
    out = cmean(x);
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

  it('compileMultiOutput', () => {
    const fn = (x: mx.array): [mx.array[], mx.array] => {
      let ys = [ x ];
      for (let i = 0; i < 5; i++) {
        ys.push(mx.add(ys[ys.length - 1], x));
      }
      return [ ys, mx.sum(ys[ys.length - 1]) ];
    }

    const x = mx.ones(1, mx.int32);
    const y1 = mx.compile(fn)(x)[1];
    const y2 = fn(x)[1];
    assert.equal(y1.item(), y2.item());
    assert.equal(y1.item(), 6);
  });

  it('infConstant', () => {
    const fn = (x) => mx.where(mx.isinf(x), 0, 1);

    const x = mx.array([0, Number.POSITIVE_INFINITY, 1], mx.bfloat16);
    assertArrayAllTrue(mx.arrayEqual(mx.compile(fn)(x), fn(x)));
  });

  it('maxIntoEqual', () => {
    const x = mx.random.uniform(0, 1, [1, 2, 2]);
    mx.eval(x);

    const fn = () => {
      const maxes = mx.max(x, [1, 2], true);
      return mx.equal(x, maxes);
    }

    const out = mx.compile(fn)();
    const expected = fn();
    assertArrayAllTrue(mx.arrayEqual(expected, out));
  });

  it('dtypes', () => {
    let x = mx.array([0, 1, 2, 3]);
    const dtypes = [mx.bool, mx.int8, mx.uint8, mx.int16, mx.uint16];
    for (const dtype of dtypes) {
      x = x.astype(dtype);
      mx.eval(x);

      const fn = (x: mx.array) => {
        return mx.add(mx.multiply(x, 1), 0);
      }

      const out = mx.compile(fn)(x);
      const expected = fn(x);
      assertArrayAllTrue(mx.arrayEqual(expected, out));
    }
  });

  it('compileDynamicDims', () => {
    const a = mx.random.uniform(0, 1, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).T;
    const b = mx.random.uniform(0, 1, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    mx.eval(a, b);

    const fn = (a: mx.array, b: mx.array) => {
      return mx.abs(mx.add(a, b));
    }

    let out = (mx.compile(fn))(a, b);
    const expected = fn(a, b);
    assert(mx.allclose(out, expected));
  });

  it('compileManyInputs', () => {
    const inputs = Array(20).fill(mx.ones([2, 2, 2, 2]));
    inputs[0] = mx.transpose(inputs[0]);

    const fun = mx.compile((...inputs: mx.array[]) => {
      let x = inputs[0];
      for (let y of inputs.slice(1, 10)) {
        x = mx.add(x, y);
      }
      let a = inputs[10];
      for (let b of inputs.slice(11)) {
        a = mx.add(a, b);
      }
      return mx.add(x, a);
    });

    const out = fun(...inputs);
    assertArrayAllTrue(mx.allclose(out, mx.full([2, 2], 20)));
  });

  it('shapelessCompileMatmul', () => {
    const a = mx.array([0.0, 1.0, 2.0]);
    const b = mx.array([0.0, 1.0, 2.0]);

    const fun = mx.compile((a: mx.array, b: mx.array) => mx.matmul(a, b), true);
    assertArrayAllTrue(mx.allclose(fun(a, b), mx.matmul(a, b)));
  });

  it('shapelessCompileSliceUpdate', () => {
    const fun = (x: mx.array) => {
      x.indexPut_(2, mx.array([3.0]));
      return x;
    };

    const cfun = mx.compile(fun, true);

    let a = mx.array([0.0, 1.0, 2.0, 3.0]);
    assertArrayAllTrue(mx.allclose(cfun(a), fun(a)));

    a = mx.array([0.0, 1.0, 2.0, 3.0, 4.0]);
    assertArrayAllTrue(mx.allclose(cfun(a), fun(a)));
  });

  it('shapelessCompileWithReshape', () => {
    const fun = (x) => {
      return mx.reshape(x, [x.shape[0] * x.shape[1], -1]);
    };

    const compiledFun = mx.compile(fun, true);

    let x = mx.zeros([2, 3, 4]);
    let out = compiledFun(x);
    assert.deepEqual(out.shape, [6, 4]);

    x = mx.zeros([2, 3, 8]);
    out = compiledFun(x);
    assert.deepEqual(out.shape, [6, 8]);

    x = mx.zeros([5, 5, 5]);

    assert.throws(() => compiledFun(x), Error);
  });

  it('compileShapelessWithBroadcast', () => {
    let a = mx.array([0.0]);
    let b = mx.ones([2, 2]);

    {
      const fun = (a: mx.array) => {
        return mx.broadcastTo(a, b.shape);
      }

      const cfun = mx.compile(fun, true);
      // Works on the first shape
      cfun(a);

      // Fails on a different shape
      assert.throws(() => {
        cfun(mx.reshape(mx.array([0.0]), [1, 1, 1]));
      });
    }

    {
      const fun = (a: mx.array, b: mx.array) => {
        return mx.broadcastArrays([a, b]);
      }

      const cfun = mx.compile(fun, true);
      [a, b] = cfun(a, b);
      assert.deepEqual(a.shape, [2, 2]);
      assert.deepEqual(b.shape, [2, 2]);
    }

    {
      // Batched matmul
      let aMatMul = mx.zeros([2, 1, 4, 2]);
      let bMatMul = mx.zeros([3, 2, 5]);

      const fun = (a: mx.array, b: mx.array) => {
        return mx.matmul(a, b);
      }

      const cfun = mx.compile(fun, true);
      let out = cfun(aMatMul, bMatMul);
      assert.deepEqual(out.shape, [2, 3, 4, 5]);
    }

    {
      // Shapeless compile should be preserved over vjp, jvp, vmap
      const fun = (args: mx.array[]) => {
        return mx.sum(mx.add(args[0], args[1]));
      }

      a = mx.array(0.0);
      b = mx.ones([2, 2]);

      const cfun = mx.compile(mx.grad(fun), true);
      let out = cfun([a, b]);

      assert.deepEqual(out[0].shape, []);
      assert.deepEqual(out[1].shape, [2, 2]);

      out = cfun([b, a]);

      assert.deepEqual(out[0].shape, [2, 2]);
      assert.deepEqual(out[1].shape, []);
    }

    {
      // Shapeless compile should be preserved over vjp, jvp, vmap
      const fun = (args: mx.array[]) => {
        return mx.sum(mx.matmul(args[0], args[1]));
      }

      let aMatMul = mx.zeros([2, 1, 4, 2]);
      let bMatMul = mx.zeros([3, 2, 5]);

      const cfun = mx.compile(mx.grad(fun), true);
      let out = cfun([aMatMul, bMatMul]);

      assert.deepEqual(out[0].shape, [2, 1, 4, 2]);
      assert.deepEqual(out[1].shape, [3, 2, 5]);

      aMatMul = mx.zeros([3, 1, 4, 2]);
      bMatMul = mx.zeros([2, 2, 5]);

      out = cfun([aMatMul, bMatMul]);

      assert.deepEqual(out[0].shape, [3, 1, 4, 2]);
      assert.deepEqual(out[1].shape, [2, 2, 5]);
    }
  });
});
