import {core as mx, optimizers as opt, nn, utils} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

const optimizersDict: Record<string, any> = getAllOptimizers();
delete optimizersDict['MultiOptimizer'];

describe('optimizers', () => {
  it('optimizerState', () => {
    const optim = new opt.SGD(0.1);
    optim.state['hello'] = 'world';
    assert.equal(optim.state['hello'], 'world');

    optim.state = {0: 1};
    assert.deepEqual(optim.state, {0: 1});
  });

  it('optimizers', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    for (let optimClass of Object.values(optimizersDict)) {
      const optim = new optimClass(0.1);
      const update = optim.applyGradients(grads, params);
      mx.eval(update);
      utils.treeMap((x: mx.array, y: mx.array) => {
        assert.deepEqual(x.shape, y.shape);
      }, params, [update]);
    }
  });

  it('typesConserved', () => {
    const params = {w: mx.ones([5, 5], mx.float16)};
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);
    for (let optimClass of Object.values(optimizersDict)) {
      const optim = new optimClass(0.1);
      const update = optim.applyGradients(grads, params);
      assert.equal(update['w'].dtype, mx.float16);
    }
  });

  it('sgd', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    // Explicit init
    let optim = new opt.SGD(0.01, 0.9);
    optim.init(params);
    utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['v'], mx.zerosLike(p)));
    }, params, [optim.state]);

    // Implicit init
    optim = new opt.SGD(0.01, 0.9);
    optim.applyGradients(grads as any, params);
    utils.treeMap((g: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['v'], g));
    }, grads, [optim.state]);
  });

  it('rmsprop', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    // Explicit init
    let optim = new opt.RMSprop(0.01);
    optim.init(params);
    utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['v'], mx.zerosLike(p)));
    }, params, [optim.state]);

    // Implicit init
    const alpha = 0.99;
    optim = new opt.RMSprop(0.01, alpha);
    optim.applyGradients(grads as any, params);
    utils.treeMap((g: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.allclose(s['v'], mx.multiply(1 - alpha, g)));
    }, grads, [optim.state]);
  });

  it('adagrad', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    // Explicit init
    let optim = new opt.Adagrad(0.01);
    optim.init(params);
    utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['v'], mx.zerosLike(p)));
    }, params, [optim.state]);
  });

  it('adadelta', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    // Explicit init
    let optim = new opt.AdaDelta(0.01);
    optim.init(params);
    utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['v'], mx.zerosLike(p)));
    }, params, [optim.state]);
    utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['u'], mx.zerosLike(p)));
    }, params, [optim.state]);
  });

  it('adam', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    // Explicit init
    const optimizers = [opt.Adam, opt.AdamW, opt.Adamax];
    for (let optimizer of optimizers) {
      const optim = new optimizer(0.01);
      optim.init(params);

      utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
        assertArrayAllTrue(mx.arrayEqual(s['v'], mx.zerosLike(p)));
        assertArrayAllTrue(mx.arrayEqual(s['m'], mx.zerosLike(p)));
      }, params, [optim.state]);
    }
  });

  it('lion', () => {
    const params = {
      'first': [mx.zeros([10]), mx.zeros([1])],
      'second': mx.zeros([1]),
    };
    const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

    // Explicit init
    let optim = new opt.Lion(0.01);
    optim.init(params);
    utils.treeMap((p: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['m'], mx.zerosLike(p)));
    }, params, [optim.state]);
  });

  it('adafactor', () => {
    let x = mx.zeros([5, 5]);
    let params = {x};
    let grad = {x: mx.onesLike(x)};
    let optimizer = new opt.Adafactor();
    for (let i = 0; i < 2; i++) {
      const xp = optimizer.applyGradients(grad, params) as {x: mx.array};
      assert.equal(xp.x.dtype, x.dtype);
      assert.deepEqual(xp.x.shape, x.shape);
    }

    x = mx.zeros([5, 5], mx.float16);
    params = {x};
    grad = {x: mx.onesLike(x)};
    optimizer = new opt.Adafactor();
    for (let i = 0; i < 2; i++) {
      const xp = optimizer.applyGradients(grad, params) as {x: mx.array};
      assert.equal(xp.x.dtype, x.dtype);
      assert.deepEqual(xp.x.shape, x.shape);
    }
    assert.equal((optimizer.state['step'] as mx.array).item(), 2);
  });

  it('compiled', function() {
    this.timeout(10 * 1000);

    const model = new nn.Linear(10, 10);
    let x = mx.random.uniform(0, 1, [2, 10]);
    let optim = new opt.SGD(0.01, 0.9);

    const origParams = model.parameters();

    function loss(model: nn.Linear, x: mx.array) {
      return model.forward(x).sum();
    }

    // Uncompiled version
    function step(x: mx.array) {
      const [_, grad] = nn.valueAndGrad(model, loss, {noAutoDispose: true})(model, x);
      optim.update(model, grad);
    }

    step(x);
    const uncompiledParams = model.parameters();

    // Pure version
    function pureLoss(params: any, x: mx.array) {
      model.update(params);
      return model.forward(x).sum();
    }

    model.update(origParams);
    optim = new opt.SGD(0.01, 0.9);

    const compiledStep = mx.compile((params: any, state: any, x: mx.array) => {
      const grad = mx.grad(pureLoss)(params, x);
      optim.state = state;
      params = optim.applyGradients(grad, params);
      return [params, optim.state];
    });

    optim.init(model.parameters());
    const [pureParams] = compiledStep(model.parameters(), optim.state, x);
    assertArrayAllTrue(mx.allclose(pureParams['weight'],
                                   uncompiledParams['weight'] as mx.array));
    assertArrayAllTrue(mx.allclose(pureParams['bias'],
                                   uncompiledParams['bias'] as mx.array));

    // TODO(zcbenz): Add impure test after implementing captures for mx.compile.
  });

  // TODO(zcbenz): Add test_update_lr_compiled after implementing captures for mx.compile.

  it('clipGradNorm', () => {
    // Test with small gradients that do not require clipping
    const smallGrads = {
      first: [mx.array([0.1, 0.2]), mx.array([0.1])],
      second: mx.array([0.3]),
    };
    let maxNorm = 10.0;  // A large maxNorm that shouldn't trigger clipping
    let [clippedGrads, totalNorm] = opt.clipGradNorm(smallGrads, maxNorm);
    utils.treeMap((x: mx.array, y: mx.array) => {
      assertArrayAllTrue(mx.arrayEqual(x, y));
    }, smallGrads, [clippedGrads]);

    // Test with large gradients that require clipping
    const largeGrads = {
      first: [mx.array([10, 20]), mx.array([10])],
      second: mx.array([30]),
    };
    maxNorm = 1.0;  // A small maxNorm that should trigger clipping
    [clippedGrads, totalNorm] = opt.clipGradNorm(largeGrads, maxNorm);
    // Correctly extract only the gradient values for norm calculation
    const clippedValues = utils.treeFlatten(clippedGrads);
    let normOfClipped = mx.array(0);
    for (const [_, g] of clippedValues) {
      normOfClipped = mx.add(normOfClipped, mx.square(g as mx.array).sum());
    }
    normOfClipped = mx.sqrt(normOfClipped);
    assert.closeTo(normOfClipped.item() as number, maxNorm, 1e-6);

    // Ensures that the scaling was done correctly
    const scale = mx.divide(maxNorm, totalNorm);
    const expectedGrads = utils.treeMap((g: mx.array) => mx.multiply(g, scale), largeGrads);
    utils.treeMap((x: mx.array, y: mx.array) => {
      assertArrayAllTrue(mx.allclose(x, y, 1e-6));
    }, expectedGrads, [clippedGrads]);
  });
});

describe('schedulers', () => {
  it('decayLr', () => {
    for (let optimClass of Object.values(optimizersDict)) {
      const lrSchedule = opt.stepDecay(1e-1, 0.9, 1);
      const optimizer = new optimClass(lrSchedule);

      const params = {w: mx.ones([5, 5])};
      const grads = utils.treeMap((x: mx.array) => mx.onesLike(x), params);

      for (let i = 0; i < 10; ++i) {
        optimizer.applyGradients(grads, params);
        assert.closeTo(optimizer.learningRate.item() as number,
                       0.1 * Math.pow(0.9, i),
                       1e-7);
      }
    }
  });

  it('stepDecay', () => {
    const lrSchedule = opt.stepDecay(1e-1, 0.9, 1000);
    const lr = lrSchedule(2500);
    const expectedLr = 0.1 * Math.pow(0.9, 2);
    assert.closeTo(lr.item() as number, expectedLr, 1e-7);
  });

  it('exponentialDecay', () => {
    const lrSchedule = opt.exponentialDecay(1e-1, 0.99);
    const lr = lrSchedule(10);
    const expectedLr = 0.1 * Math.pow(0.99, 10);
    assert.closeTo(lr.item() as number, expectedLr, 1e-7);
  });

  it('cosineDecay', () => {
    let lrSchedule = opt.cosineDecay(0.1, 10);
    let lr = lrSchedule(4);
    let expectedLr = 0.1 * 0.5 * (1.0 + Math.cos(Math.PI * 4 / 10));
    assert.closeTo(lr.item() as number, expectedLr, 1e-7);

    lrSchedule = opt.cosineDecay(0.1, 10, 0.05);
    lr = lrSchedule(9);
    const expectedEndLr = 0.05;
    assert.isAbove(lr.item() as number, expectedEndLr);
    lr = lrSchedule(20);
    assert.closeTo(lr.item() as number, expectedEndLr, 1e-7);
  });

  it('scheduleJoiner', () => {
    let boundaries = [2, 3, 4];
    let schedules = [() => 3, () => 4, () => 5];
    assert.throws(() => opt.joinSchedules(schedules, boundaries), Error);
    boundaries = [2, 4];
    const schedule = opt.joinSchedules(schedules, boundaries);
    assert.equal(schedule(0).item(), 3);
    assert.equal(schedule(1).item(), 3);
    assert.equal(schedule(2).item(), 4);
    assert.equal(schedule(3).item(), 4);
    assert.equal(schedule(5).item(), 5);
    assert.equal(schedule(7).item(), 5);
  });

  it('linearWarmupWithCosineDecay', () => {
    const warmupSchedule = opt.linearSchedule(0.0, 1e-5, 100);
    const cosineSchedule = opt.cosineDecay(1e-5, 100);
    const cosWithWarmup = opt.joinSchedules([warmupSchedule, cosineSchedule], [101]);
    assert.equal(cosWithWarmup(0).item(), 0.0);
    assert.closeTo(cosWithWarmup(101).item() as number, 1e-5, 0.1);
    const optimizer = new opt.Adam(cosWithWarmup);
    const model = new nn.Linear(10, 10);
    for (let i = 0; i < 100; ++i) {
      optimizer.update(model, {});
    }
    assert.closeTo(optimizer.learningRate.item() as number, 1e-5, 0.1);
    for (let i = 0; i < 100; ++i) {
      optimizer.update(model, {});
    }
    const expectedLr = 1e-5 * 0.5 * (1.0 + Math.cos(Math.PI * 200 / 10));
    assert.closeTo(optimizer.learningRate.item() as number, expectedLr, 0.1);
  });

  // TODO(zcbenz): Add test_compile_with_schedule after implementing captures for mx.compile.

  it('initFromState', () => {
    class Model extends nn.Module {
      constructor() {
        super();
        this.l1 = new nn.Linear(2, 2);
        this.drop = new nn.Dropout(0.5);
        this.l2 = new nn.Linear(2, 2);
        this.vals = [new nn.Linear(2, 2), new nn.ReLU(), new nn.ReLU()];
      }

      override forward(x: mx.array) {
        return x;
      }
    }

    const model = new Model();
    let optimizer = new opt.Adam(3e-4);
    optimizer.init(model.trainableParameters());

    // Flatten the state for serialization
    const state = utils.treeFlatten(optimizer.state);

    // Make a new optimizer and load the state
    optimizer = new opt.Adam(3e-4);
    optimizer.state = utils.treeUnflatten(state) as typeof optimizer.state;

    // This should work without any errors
    const grads = model.trainableParameters();
    optimizer.update(model, grads);
  });

  it('multiOptimizer', () => {
    class Model extends nn.Module {
      l1: nn.Linear;
      drop: nn.Dropout;
      l2: nn.Linear;
      vals: [nn.Linear, nn.ReLU, nn.ReLU];

      constructor() {
        super();
        this.l1 = new nn.Linear(2, 2);
        this.drop = new nn.Dropout(0.5);
        this.l2 = new nn.Linear(2, 2);
        this.vals = [new nn.Linear(2, 2), new nn.ReLU(), new nn.ReLU()];
      }

      override forward(x: mx.array) {
        return x;
      }
    }

    const model = new Model();
    const optimizer = new opt.MultiOptimizer(
      [new opt.Adam(0.001), new opt.SGD(0.1)],
      [(name: string, weight: mx.array) => weight.shape.length > 1],
    );
    optimizer.init(model.trainableParameters());

    assert.equal(Object.keys(optimizer.state.states).length, 2);

    const adamStates = utils.treeFlatten(optimizer.state.states[0]);
    const sgdStates = utils.treeFlatten(optimizer.state.states[1]);
    assert.equal((sgdStates.length - 2) * 2, adamStates.length - 2);
    assert.isFalse(adamStates.some(([k, v]) => k.includes('bias')));
    assert.isFalse(sgdStates.some(([k, v]) => k.includes('weight')));
  });
});

function getAllOptimizers(): Record<string, any> {
  const classes: Record<string, any> = {};
  for (const [name, obj] of Object.entries(opt)) {
    if (obj.prototype instanceof opt.Optimizer)
      classes[name] = obj;
  }
  return classes;
}
