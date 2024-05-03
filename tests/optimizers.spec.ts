import {core as mx, optimizers as opt, nn, utils} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

const optimizersDict: Record<string, any> = getAllOptimizers();

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
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
    optim.applyGradients(grads as Record<string, unknown>, params);
    utils.treeMap((g: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.arrayEqual(s['v'], g));
    }, grads, [optim.state]);
  });

  it('rmsprop', () => {
    const params = {
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
    optim.applyGradients(grads as Record<string, unknown>, params);
    utils.treeMap((g: mx.array, s: Record<string, mx.array>) => {
      assertArrayAllTrue(mx.allclose(s['v'], mx.multiply(1 - alpha, g)));
    }, grads, [optim.state]);
  });

  it('adagrad', () => {
    const params = {
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
      first: [mx.zeros([10]), mx.zeros([1])],
      second: mx.zeros([1]),
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
    let grad = mx.onesLike(x);
    let optimizer = new opt.Adafactor();
    for (let i = 0; i < 2; i++) {
      const xp = optimizer.applyGradients(grad, x);
      assert.equal(xp.dtype, x.dtype);
      assert.deepEqual(xp.shape, x.shape);
    }

    x = mx.zeros([5, 5], mx.float16);
    grad = mx.onesLike(x);
    optimizer = new opt.Adafactor();
    for (let i = 0; i < 2; i++) {
      const xp = optimizer.applyGradients(grad, x);
      assert.equal(xp.dtype, x.dtype);
      assert.deepEqual(xp.shape, x.shape);
    }
    assert.equal((optimizer.state['step'] as mx.array).item(), 2);
  });

  it('compiled', () => {
    const model = new nn.Linear(10, 10);
    let x = mx.random.uniform(0, 1, [2, 10]);
    let optim = new opt.SGD(0.01, 0.9);

    const origParams = model.parameters();

    function loss(model: nn.Linear, x: mx.array) {
      return model.forward(x).sum();
    }

    // Uncompiled version
    function step(x: mx.array) {
      const [_, grad] = nn.valueAndGrad(model, loss)(model, x);
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
});

function getAllOptimizers(): Record<string, any> {
  const classes: Record<string, any> = {};
  for (const [name, obj] of Object.entries(opt)) {
    if (obj.prototype instanceof opt.Optimizer)
      classes[name] = obj;
  }
  return classes;
}
