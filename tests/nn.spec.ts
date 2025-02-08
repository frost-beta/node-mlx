import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {core as mx, nn, utils} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('base', () => {
  it('moduleUtilities', () => {
    const m = new nn.Sequential(
      new nn.Sequential(new nn.Linear(2, 10), nn.relu),
      new nn.Sequential(new nn.Linear(10, 10), new nn.ReLU()),
      new nn.Linear(10, 1),
      mx.sigmoid,
    );

    const children = m.children();
    assert.isTrue(typeof children === 'object');
    assert.equal(Object.keys(children).length, 1);
    const layers = children['layers'];
    assert.isTrue(Array.isArray(layers));
    assert.equal(layers.length, 4);
    assert.deepEqual(layers[3], {});
    const flatChildren = utils.treeFlatten(children, '', nn.Module.isModule);
    assert.equal(flatChildren.length, 3);

    const leaves = utils.treeFlatten(m.leafModules(), '', nn.Module.isModule);
    assert.equal(leaves.length, 4);
    assert.equal(leaves[0][0], 'layers.0.layers.0');
    assert.equal(leaves[1][0], 'layers.1.layers.0');
    assert.equal(leaves[2][0], 'layers.1.layers.1');
    assert.equal(leaves[3][0], 'layers.2');
    assert.isTrue(leaves[0][1] === (m.layers[0] as nn.Sequential).layers[0]);
    assert.isTrue(leaves[1][1] === (m.layers[1] as nn.Sequential).layers[0]);
    assert.isTrue(leaves[2][1] === (m.layers[1] as nn.Sequential).layers[1]);
    assert.isTrue(leaves[3][1] === m.layers[2]);

    m.eval();

    m.applyToModules((k, m) => {
      assert.isFalse(m.training);
    });

    m.train();

    m.applyToModules((k, m) => {
      assert.isTrue(m.training);
    });
  });

  it('moduleAttributes', () => {
    class Model extends nn.Module {
      val: mx.array | null;

      constructor() {
        super();
        this.val = null;
        this.initialize();
      }

      initialize() {
        this.val = mx.array(1.0);
      }

      forward() {
      }
    }

    const model = new Model();
    assertArrayAllTrue(mx.arrayEqual(model.val!, mx.array(1.0)));

    model.val = null;
    assert.strictEqual(model.val, null)

    model.val = mx.array([3])
    assert.equal(model.val!.item(), 3);
  });

  it('modelWithDict', () => {
    class DictModule extends nn.Module {
      weights: {w1: mx.array, w2: mx.array};

      constructor() {
        super();
        this.weights = {w1: mx.zeros([2, 2]), w2: mx.ones([2, 2])};
      }

      forward() {
      }
    }

    const model = new DictModule();
    const params = model.parameters();
    assert.equal(utils.treeFlatten(params).length, 2);
    assertArrayAllTrue(mx.arrayEqual(params['weights']['w1'], mx.zeros([2, 2])));
    assertArrayAllTrue(mx.arrayEqual(params['weights']['w2'], mx.ones([2, 2])));
  });

  it('saveSafetensorsWeights', () => {
    const makeModel = () => {
      return new nn.Sequential(new nn.Linear(2, 2),
                               new nn.ReLU(),
                               new nn.Linear(2, 2),
                               new nn.ReLU());
    };

    const tdir = fs.mkdtempSync(path.join(os.tmpdir(), 'test-'));
    const safetensorsFile = path.join(tdir, 'model.safetensors');
    after(() => fs.rmSync(tdir, { recursive: true }));

    const m = makeModel();
    const params = Object.fromEntries(utils.treeFlatten(m.parameters())) as Record<string, mx.array>;
    m.saveWeights(safetensorsFile);

    const mLoad = makeModel();
    mLoad.loadWeights(safetensorsFile);

    // Eval so the model file is not held.
    mx.eval(mLoad);

    const eqTree = utils.treeMap(mx.arrayEqual, m.parameters(), [ mLoad.parameters() ]);
    assert.notInclude(utils.treeFlatten(eqTree).map(v => v[1]), false);
  });

  it('loadFromWeights', () => {
    const m = new nn.Linear(2, 2);

    // Too few weights
    let weights: [string, mx.array][] = [['weight', mx.ones([2, 2])]];
    assert.throws(() => m.loadWeights(weights), Error);

    m.loadWeights(weights, false);
    assertArrayAllTrue(mx.arrayEqual(m.weight, weights[0][1]));

    // Wrong name
    assert.throws(() => m.loadWeights([['weihgt', mx.ones([2, 2])]]), Error);

    // Ok
    m.loadWeights([['weihgt', mx.ones([2, 2])]], false);

    // Too many weights
    assert.throws(() => m.loadWeights([
      ['weight', mx.ones([2, 2])],
      ['bias', mx.ones([2])],
      ['bias2', mx.ones([2])]
    ]), Error);

    // Wrong shape
    assert.throws(() => m.loadWeights([
      ['weight', mx.ones([2, 2])],
      ['bias', mx.ones([2, 1])]
    ]), Error);

    // Wrong type
    assert.throws(() => m.loadWeights([
      ['weight', mx.ones([2, 2])],
      ['bias', 3 as unknown as mx.array],
    ]), Error);
  });

  it('snakeWeights', () => {
    class Model extends nn.Module {
      someWeight: mx.array;

      constructor() {
        super();
        this.someWeight = mx.zeros([2, 2]);
      }

      override forward() {}
    }

    const m = new Model();
    m.loadWeights([ ['some_weight', mx.ones([2, 2])] ]);
    assertArrayAllTrue(mx.arrayEqual(m.someWeight, mx.ones([2, 2])));

    class NestedModel extends nn.Module {
      someChild: Model;

      constructor() {
        super();
        this.someChild = new Model();
      }

      override forward() {}
    }

    const n = new NestedModel();
    n.loadWeights([ ['some_child.some_weight', mx.ones([2, 2])] ]);
    assertArrayAllTrue(mx.arrayEqual(n.someChild.someWeight, mx.ones([2, 2])));
  });

  it('moduleState', () => {
    const m = new nn.Linear(10, 1);
    m.state['hello'] = 'world';
    assert.equal(m.state['hello'], 'world');
  });

  it('chaining', () => {
    const m = new nn.Sequential(new nn.Linear(2, 2), new nn.ReLU(), new nn.Linear(2, 1));
    const preFreezeNumParams = Object.keys(m.parameters()).length;
    m.freeze().unfreeze();
    assert.equal(Object.keys(m.parameters()).length, preFreezeNumParams);
    const paramsDict = m.parameters();

    assert.isFalse(m.update(paramsDict).eval().training);
    assert.isTrue(m.train().training);
  });

  it('quantize', () => {
    let m = new nn.Sequential(new nn.Embedding(5, 256), new nn.ReLU(), new nn.Linear(256, 256));
    nn.quantize(m);
    assert.isTrue(m.layers[0] instanceof nn.QuantizedEmbedding);
    assert.isTrue(m.layers[1] instanceof nn.ReLU);
    assert.isTrue(m.layers[2] instanceof nn.QuantizedLinear);

    m = new nn.Sequential(new nn.Embedding(5, 256), new nn.ReLU(), new nn.Linear(256, 256));
    nn.quantize(m, undefined, undefined, (_, m: nn.Module) => m instanceof nn.Linear);
    assert.isTrue(m.layers[0] instanceof nn.Embedding);
    assert.isTrue(m.layers[1] instanceof nn.ReLU);
    assert.isTrue(m.layers[2] instanceof nn.QuantizedLinear);
  });

  // FIXME(zcbenz): Port 33421c1dd from MLX.
  xit('gradOfModule', () => {
    class Model extends nn.Module {
      m1: nn.Linear;
      constructor() {
        super();
        this.m1 = new nn.Linear(3, 3);
      }
      override forward(x: mx.array) {
        return this.m1.forward(x);
      }
    }

    const model = new Model();

    const lossFn = (model: Model) => model.forward(x);

    const x = mx.zeros([3]);
    mx.grad(lossFn)(model);
  });
});

describe('layers', function() {
  this.timeout(10 * 1000);

  it('identity', () => {
    const inputs = mx.zeros([10, 4]);
    const layer = new nn.Identity();
    const outputs = layer.forward(inputs);
    assert.deepEqual(inputs.shape, outputs.shape);
  });

  it('linear', () => {
    const inputs = mx.zeros([10, 4]);
    const layer = new nn.Linear(4, 8);
    const outputs = layer.forward(inputs);
    assert.deepEqual(outputs.shape, [10, 8]);
  });

  it('bilinear', () => {
    const inputs1 = mx.zeros([10, 2]);
    const inputs2 = mx.zeros([10, 4]);
    const layer = new nn.Bilinear(2, 4, 6);
    const outputs = layer.forward(inputs1, inputs2);
    assert.deepEqual(outputs.shape, [10, 6]);
  });

  it('groupNorm', () => {
    let x = mx.arange(100, mx.float32)
    x = x.reshape(1, 10, 10, 1);
    x = mx.broadcastTo(x, [2, 10, 10, 4]);
    x = mx.concatenate([x, mx.multiply(x, 0.5)], -1);

    // Group norm in groups last mode.
    let g = new nn.GroupNorm(2, 8);
    let y = g.forward(x);
    let means = y.reshape(2, -1, 2).mean(1);
    let variances = y.reshape(2, -1, 2).variance(1);
    assertArrayAllTrue(mx.allclose(means,
                                   mx.zerosLike(means),
                                   undefined, 1e-6));
    assertArrayAllTrue(mx.allclose(variances,
                                   mx.onesLike(variances),
                                   undefined, 1e-6));

    g.weight = mx.multiply(g.weight!, 2);
    g.bias = mx.add(g.bias!, 3);
    y = g.forward(x);
    means = y.reshape(2, -1, 2).mean(1);
    variances = y.reshape(2, -1, 2).variance(1);
    assertArrayAllTrue(mx.allclose(means,
                                   mx.multiply(mx.onesLike(means), 3),
                                   undefined, 1e-6));
    assertArrayAllTrue(mx.allclose(variances,
                                   mx.multiply(mx.onesLike(variances), 4),
                                   undefined, 1e-6));

    // Group norm in groups first mode.
    g = new nn.GroupNorm(2, 8, undefined, undefined, true);
    y = g.forward(x);
    means = y.reshape(2, -1, 2, 4).mean([1, -1]);
    variances = y.reshape(2, -1, 2, 4).variance([1, -1]);
    assertArrayAllTrue(mx.allclose(means,
                                   mx.zerosLike(means),
                                   undefined, 1e-6));
    assertArrayAllTrue(mx.allclose(variances,
                                   mx.onesLike(variances),
                                   undefined, 1e-6));

    g.weight = mx.multiply(g.weight!, 2);
    g.bias = mx.add(g.bias!, 3);
    y = g.forward(x);
    means = y.reshape(2, -1, 2, 4).mean([1, -1]);
    variances = y.reshape(2, -1, 2, 4).variance([1, -1]);
    assertArrayAllTrue(mx.allclose(means,
                                   mx.multiply(mx.onesLike(means), 3),
                                   undefined, 1e-6));
    assertArrayAllTrue(mx.allclose(variances,
                                   mx.multiply(mx.onesLike(variances), 4),
                                   undefined, 1e-6));
  });

  it('instanceNorm', () => {
    // Test InstanceNorm1d.
    let x = mx.array([
      [
        [-0.0119524, 1.1263, 2.02223],
        [-0.500331, 0.517899, -1.21143],
        [1.12958, -0.21413, -2.48738],
        [1.39955, 0.891329, 1.63289]
      ],
      [
        [0.241417, -0.619157, -0.77484],
        [-1.42512, 0.970817, -1.31352],
        [2.739, -1.2506, 1.56844],
        [-1.23175, 0.32756, 1.13969]
      ]
    ]);
    let inorm = new nn.InstanceNorm(3);
    let y = inorm.forward(x);
    const expectedY = mx.array([
      [
        [-0.657082, 1.07593, 1.0712],
        [-1.27879, -0.123074, -0.632505],
        [0.796101, -1.56572, -1.30476],
        [1.13978, 0.612862, 0.866067]
      ],
      [
        [0.0964426, -0.557906, -0.759885],
        [-0.904772, 1.30444, -1.20013],
        [1.59693, -1.29752, 1.15521],
        [-0.7886, 0.550987, 0.804807]
      ]
    ]);
    assert.deepEqual(x.shape, y.shape);
    assertArrayAllTrue(mx.allclose(y, expectedY, undefined, 1e-5));

    // Test InstanceNorm2d.
    x = mx.array([
      [
        [
          [-0.458824, 0.483254, -0.58611],
          [-0.447996, -0.176577, -0.622545],
          [0.0486988, -0.0611224, 1.8845]
        ],
        [
          [1.13049, 0.345315, -0.926389],
          [0.301795, 0.99207, -0.184927],
          [-2.23876, -0.758631, -1.12639]
        ],
        [
          [0.0986325, -1.82973, -0.241765],
          [-1.25257, 0.154442, -0.556204],
          [-0.329399, -0.319107, 0.830584]
        ]
      ],
      [
        [
          [1.04407, 0.073752, 0.407081],
          [0.0800776, 1.2513, 1.20627],
          [0.782321, -0.444367, 0.563132]
        ],
        [
          [0.671423, -1.21689, -1.88979],
          [-0.110299, -1.42248, 1.17838],
          [0.159905, 0.516452, -0.539121]
        ],
        [
          [0.810252, 1.50456, 1.08659],
          [0.182597, 0.0576239, 0.973883],
          [-0.0621687, 0.184253, 0.784216]
        ]
      ]
    ]);
    inorm = new nn.InstanceNorm(3);
    y = inorm.forward(x);
    const expectedY2 = mx.array([
      [
        [
          [-0.120422, 0.801503, -0.463983],
          [-0.108465, -0.0608611, -0.504602],
          [0.440008, 0.090032, 2.29032]
        ],
        [
          [1.63457, 0.621224, -0.843335],
          [0.719488, 1.4665, -0.0167344],
          [-2.08591, -0.821575, -1.0663]
        ],
        [
          [0.495147, -2.22145, -0.0800989],
          [-0.996913, 0.371763, -0.430643],
          [0.022495, -0.24714, 1.11538]
        ]
      ],
      [
        [
          [1.5975, 0.0190292, -0.0123306],
          [-0.776381, 1.28291, 0.817237],
          [0.952927, -0.537076, 0.149652]
        ],
        [
          [0.679836, -1.36624, -2.39651],
          [-1.24519, -1.5869, 0.788287],
          [-0.579802, 0.494186, -0.994499]
        ],
        [
          [1.02171, 1.55474, 0.693008],
          [-0.523922, 0.00171862, 0.576016],
          [-1.12667, 0.137632, 0.37914]
        ]
      ]
    ]);
    assert.deepEqual(x.shape, y.shape);
    assertArrayAllTrue(mx.allclose(y, expectedY2, undefined, 1e-5));

    // Test repr.
    assert.equal(inorm.toString(), 'InstanceNorm(3, eps=1e-5, affine=false)');
  });

  it('batchNorm', () => {
    mx.random.seed(42);
    let x = mx.random.normal([5, 4]);

    // Batch norm.
    let bn = new nn.BatchNorm(4, undefined, undefined, true);
    assertArrayAllTrue(mx.equal(bn.runningMean!, mx.zerosLike(bn.runningMean!)));
    assertArrayAllTrue(mx.equal(bn.runningVar!, mx.onesLike(bn.runningVar!)));
    let y = bn.forward(x);
    let expectedY = mx.array([
      [-0.439520, 1.647328, -0.955515, 1.966031],
      [-1.726690, -1.449826, -0.234026, -0.723364],
      [0.938414, -0.349603, -0.354470, -0.175369],
      [0.305006, 0.234914, -0.393017, -0.459385],
      [0.922789, -0.082813, 1.937028, -0.607913],
    ]);
    let expectedMean = mx.array([0.008929, 0.005680, -0.016092, 0.027778]);
    let expectedVar = mx.array([0.928435, 1.00455, 1.04117, 0.94258]);
    assert.deepEqual(x.shape, y.shape);
    assertArrayAllTrue(mx.allclose(y, expectedY, undefined, 1e-5));
    assertArrayAllTrue(mx.allclose(bn.runningMean!, expectedMean, undefined, 1e-5));
    assertArrayAllTrue(mx.allclose(bn.runningVar!, expectedVar, undefined, 1e-5));

    // test eval mode.
    bn.eval();
    y = bn.forward(x);
    expectedY = mx.array([
      [-0.15984, 1.73159, -1.25456, 1.57891],
      [-0.872193, -1.4281, -0.414439, -0.228678],
      [0.602743, -0.30566, -0.554687, 0.139639],
      [0.252199, 0.29066, -0.599572, -0.0512532],
      [0.594096, -0.0334829, 2.11359, -0.151081],
    ]);
    assert.deepEqual(x.shape, y.shape);
    assertArrayAllTrue(mx.allclose(y, expectedY, undefined, 1e-5));

    // test_no_affine.
    bn = new nn.BatchNorm(4, undefined, undefined, false);
    y = bn.forward(x);
    expectedY = mx.array([
      [-0.439520, 1.647328, -0.955515, 1.966031],
      [-1.726690, -1.449826, -0.234026, -0.723364],
      [0.938414, -0.349603, -0.354470, -0.175369],
      [0.305006, 0.234914, -0.393017, -0.459385],
      [0.922789, -0.082813, 1.937028, -0.607913],
    ]);
    assert.deepEqual(x.shape, y.shape);
    assertArrayAllTrue(mx.allclose(y, expectedY, undefined, 1e-5));

    // test with 3D input.
    mx.random.seed(42);
    const N = 2;
    const L = 4;
    const C = 5;
    x = mx.random.normal([N, L, C]);

    // Batch norm.
    bn = new nn.BatchNorm(C, undefined, undefined, true);
    assertArrayAllTrue(mx.equal(bn.runningMean!, mx.zerosLike(bn.runningMean!)));
    assertArrayAllTrue(mx.equal(bn.runningVar!, mx.onesLike(bn.runningVar!)));
    y = bn.forward(x);
    assert.deepEqual(x.shape, y.shape);
    expectedY = mx.array([
      [
        [-0.335754, 0.342054, 1.02653, 0.628588, -1.63899],
        [1.92092, 0.432319, 0.343043, 1.95489, 1.0696],
        [-0.853748, 1.3661, 0.868569, 0.0199196, -0.887284],
        [0.459206, -0.684822, -0.706354, -0.271531, 0.566341],
      ],
      [
        [-0.921179, 0.684951, -0.77466, -0.490372, -0.247032],
        [1.10839, -2.13179, 0.628924, -1.62639, -0.539708],
        [-0.348943, 0.412194, -2.03818, 0.524972, 1.64568],
        [-1.02889, -0.421, 0.652127, -0.740079, 0.0313996],
      ],
    ]);
    assertArrayAllTrue(mx.allclose(y, expectedY, undefined, 1e-5));
    expectedMean = mx.array([[[0.00207845, -5.3259e-05, 0.04755, -0.0697296, 0.0236228]]]);
    expectedVar = mx.array([[[0.968415, 1.05322, 0.96913, 0.932305, 0.967224]]]);
    assertArrayAllTrue(mx.allclose(bn.runningMean!, expectedMean, undefined, 1e-5));
    assertArrayAllTrue(mx.allclose(bn.runningVar!, expectedVar, undefined, 1e-5));

    x = mx.random.normal([N, L, C, L, C]);
    assert.throws(() => { bn.forward(x) }, Error);

    // Check that the running stats are in the param dictionary.
    const bnParameters = bn.parameters();
    assert.property(bnParameters, 'runningMean');
    assert.property(bnParameters, 'runningVar');
    assert.property(bnParameters, 'weight');
    assert.property(bnParameters, 'bias');

    let bnTrainable = bn.trainableParameters();
    assert.notProperty(bnTrainable, 'runningMean');
    assert.notProperty(bnTrainable, 'runningVar');
    assert.property(bnTrainable, 'weight');
    assert.property(bnTrainable, 'bias');

    bn.unfreeze();
    bnTrainable = bn.trainableParameters();
    assert.notProperty(bnTrainable, 'runningMean');
    assert.notProperty(bnTrainable, 'runningVar');
    assert.property(bnTrainable, 'weight');
    assert.property(bnTrainable, 'bias');
  });

  it('batchNormStats', () => {
    const batchSize = 2;
    const numFeatures = 4;
    const h = 3;
    const w = 3;
    const momentum = 0.1;

    let batchNorm = new nn.BatchNorm(numFeatures);

    batchNorm.train();
    let runningMean = batchNorm.runningMean!;
    let runningVar = batchNorm.runningVar!;

    let data = mx.random.normal([batchSize, numFeatures]);
    let normalizedData = batchNorm.forward(data);
    let means = data.mean(0);
    let variances = data.variance(0);
    runningMean = mx.add(mx.multiply(runningMean, (1 - momentum)), mx.multiply(means, momentum));
    runningVar = mx.add(mx.multiply(runningVar, (1 - momentum)), mx.multiply(variances, momentum));
    assertArrayAllTrue(mx.allclose(batchNorm.runningMean!, runningMean, 1e-5));
    assertArrayAllTrue(mx.allclose(batchNorm.runningVar!, runningVar, 1e-5));

    batchNorm = new nn.BatchNorm(numFeatures);
    batchNorm.train();
    runningMean = batchNorm.runningMean!;
    runningVar = batchNorm.runningVar!;
    data = mx.random.normal([batchSize, h, w, numFeatures]);

    normalizedData = batchNorm.forward(data);
    means = data.mean([0, 1, 2]);
    variances = data.variance([0, 1, 2]);
    runningMean = mx.add(mx.multiply(runningMean, (1 - momentum)), mx.multiply(means, momentum));
    runningVar = mx.add(mx.multiply(runningVar, (1 - momentum)), mx.multiply(variances, momentum));
    assertArrayAllTrue(mx.allclose(batchNorm.runningMean!, runningMean, 1e-5));
    assertArrayAllTrue(mx.allclose(batchNorm.runningVar!, runningVar, 1e-5));

    assert.deepEqual(batchNorm.runningMean!.shape, runningMean.shape);
    assert.deepEqual(batchNorm.runningVar!.shape, runningVar.shape);
  });

  it('conv1d', () => {
    const N = 5;
    const L = 12;
    const ks = 3;
    const CIn = 2;
    const COut = 4;
    let x = mx.ones([N, L, CIn]);
    let c = new nn.Conv1d(CIn, COut, ks);
    c.weight = mx.onesLike(c.weight);
    let y = c.forward(x);
    assert.deepEqual(y.shape, [N, L - ks + 1, COut]);
    assertArrayAllTrue(mx.arrayEqual(y, mx.full(y.shape, ks * CIn)));

    c = new nn.Conv1d(CIn, COut, ks, 2);
    y = c.forward(x);
    assert.deepEqual(y.shape, [N, Math.floor((L - ks + 1) / 2), COut]);
    assert.property(c.parameters(), 'bias');

    const dil = 2;
    c = new nn.Conv1d(CIn, COut, ks, undefined, undefined, dil);
    y = c.forward(x);
    assert.deepEqual(y.shape, [N, L - (ks - 1) * dil, COut]);

    c = new nn.Conv1d(CIn, COut, ks, undefined, undefined, undefined, undefined, false);
    assert.notProperty(c.parameters(), 'bias');

    const groups = CIn;
    c = new nn.Conv1d(CIn, COut, ks, undefined, undefined, undefined, groups);
    y = c.forward(x);
    assert.deepEqual(c.weight.shape, [COut, ks, CIn / groups]);
    assert.deepEqual(y.shape, [N, L - ks + 1, COut]);
  });

  it('conv2d', () => {
    let x = mx.ones([4, 8, 8, 3]);
    let c = new nn.Conv2d(3, 1, 8);
    let y = c.forward(x);
    assert.deepEqual(y.shape, [4, 1, 1, 1]);
    c.weight = mx.divide(mx.onesLike(c.weight), 8*8*3);
    y = c.forward(x);
    assertArrayAllTrue(mx.allclose(y.index(0, 0, 0), x.mean([1, 2, 3])));

    // 3x3 conv no padding stride 1
    c = new nn.Conv2d(3, 8, 3);
    y = c.forward(x);
    assert.deepEqual(y.shape, [4, 6, 6, 8]);
    assert.isBelow(mx.subtract(y, c.weight.sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);

    // 3x3 conv padding 1 stride 1
    c = new nn.Conv2d(3, 8, 3, 1, 1);
    y = c.forward(x);
    assert.deepEqual(y.shape, [4, 8, 8, 8]);
    assert.isBelow(mx.subtract(y.index(mx.Slice(), mx.Slice(1, 7), mx.Slice(1, 7)),
                               c.weight.sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);
    assert.isBelow(mx.subtract(y.index(mx.Slice(), 0, 0),
                               c.weight.index(mx.Slice(), mx.Slice(1), mx.Slice(1))
                                .sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);
    assert.isBelow(mx.subtract(y.index(mx.Slice(), 7, 7),
                               c.weight.index(mx.Slice(), mx.Slice(null, -1), mx.Slice(null, -1))
                                .sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);
    assert.isBelow(mx.subtract(y.index(mx.Slice(), mx.Slice(1, 7), 7),
                               c.weight.index(mx.Slice(), mx.Slice(), mx.Slice(null, -1))
                                .sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);
    assert.isBelow(mx.subtract(y.index(mx.Slice(), 7, mx.Slice(1, 7)),
                               c.weight.index(mx.Slice(), mx.Slice(null, -1), mx.Slice())
                                .sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);

    // 3x3 conv no padding stride 2
    c = new nn.Conv2d(3, 8, 3, 2, 0);
    y = c.forward(x);
    assert.deepEqual(y.shape, [4, 3, 3, 8]);
    assert.isBelow(mx.subtract(y, c.weight.sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);

    c = new nn.Conv2d(3, 8, 3, undefined, undefined, 2);
    y = c.forward(x);
    assert.deepEqual(y.shape, [4, 4, 4, 8]);
    assert.isBelow(mx.subtract(y, c.weight.sum([1, 2, 3]))
                     .abs().max().item() as number,
                   1e-4);

    // 3x3 conv groups > 1
    x = mx.ones([4, 7, 7, 4]);
    c = new nn.Conv2d(4, 8, 3, 1, 1, undefined, 2);
    y = c.forward(x);
    assert.deepEqual(y.shape, [4, 7, 7, 8]);
  });

  it('sequential', () => {
    const x = mx.ones([10, 2]);
    const m = new nn.Sequential(new nn.Linear(2, 10), new nn.ReLU(), new nn.Linear(10, 1));
    const y = m.forward(x);
    assert.deepEqual(y.shape, [10, 1]);
    const params = m.parameters();
    assert.property(params, 'layers');
    assert.equal(params['layers'].length, 3);
    assert.property(params['layers'][0], 'weight');
    assert.equal(Object.keys(params['layers'][1]).length, 0);
    assert.property(params['layers'][2], 'weight');

    m.layers[1] = nn.relu;
    const y2 = m.forward(x);
    assertArrayAllTrue(mx.arrayEqual(y, y2));
  });

  it('gelu', function() {
    this.timeout(20 * 1000);
    const inputs = mx.array([1.15286231, -0.81037411, 0.35816911, 0.77484438, 0.66276414]);
    const expected = mx.array([1.0093501, -0.16925684, 0.22918941, 0.60498625, 0.49459383]);
    const expectedApprox = mx.array([1.0091482, -0.1693441, 0.22918446, 0.60491, 0.4945476]);

    const out = new nn.GELU().forward(inputs);
    assertArrayAllTrue(mx.isclose(out, expected));

    const outApprox = new nn.GELU('precise').forward(inputs);
    assertArrayAllTrue(mx.isclose(outApprox, expectedApprox));

    const x = mx.arange(-6.0, 6.0, 12 / 100);
    const y = nn.gelu(x);
    const yHat1 = nn.geluApprox(x);
    const yHat2 = nn.geluFastApprox(x);

    assert.isBelow(mx.subtract(y, yHat1).abs().max().item() as number, 0.0005);
    assert.isBelow(mx.subtract(y, yHat2).abs().max().item() as number, 0.025);
  });

  it('sinPe', () => {
    const m = new nn.SinusoidalPositionalEncoding(16, 0.01);
    const x = mx.arange(10);
    const y = m.forward(x);
    assert.deepEqual(y.shape, [10, 16]);

    const similarities = mx.matmul(y, y.T);
    assert.isBelow(mx.subtract(similarities.index(mx.arange(10, mx.int32),
                                                  mx.arange(10, mx.int32)),
                               1)
                     .abs().max().item() as number,
                   1e-5);
  });

  it('sigmoid', () => {
    const x = mx.array([1.0, 0.0, -1.0]);
    const y1 = mx.sigmoid(x);
    const y2 = nn.sigmoid(x);
    const y3 = (new nn.Sigmoid()).forward(x);

    assertArrayAllTrue(mx.equal(y1, y2));
    assertArrayAllTrue(mx.equal(y1, y3));
  });

  it('relu', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    const y = nn.relu(x);
    assertArrayAllTrue(mx.arrayEqual(y, mx.array([1.0, 0.0, 0.0])));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('leakyRelu', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    let y = nn.leakyRelu(x);
    assertArrayAllTrue(mx.arrayEqual(y, mx.array([1.0, -0.01, 0.0])));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);

    y = new nn.LeakyReLU(0.1).forward(x);
    assertArrayAllTrue(mx.arrayEqual(y, mx.array([1.0, -0.1, 0.0])));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('elu', () => {
    let x = mx.array([1.0, -1.0, 0.0]);
    let y = nn.elu(x);
    const epsilon = 1e-4;
    let expectedY = mx.array([1.0, -0.6321, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);

    y = new nn.ELU(1.1).forward(x);
    expectedY = mx.array([1.0, -0.6953, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('relu6', () => {
    const x = mx.array([1.0, -1.0, 0.0, 7.0, -7.0]);
    const y = nn.relu6(x);
    assertArrayAllTrue(mx.arrayEqual(y, mx.array([1.0, 0.0, 0.0, 6.0, 0.0])));
    assert.deepEqual(y.shape, [5]);
    assert.equal(y.dtype, mx.float32);
  });

  it('softmax', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    const y = nn.softmax(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([0.6652, 0.0900, 0.2447]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('softmin', () => {
    const x = mx.array([1.0, 2.0, 3.0]);
    const y = nn.softmin(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([0.6652, 0.2447, 0.0900]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('softplus', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    const y = nn.softplus(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([1.3133, 0.3133, 0.6931]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('softsign', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    const y = nn.softsign(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([0.5, -0.5, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('softshrink', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    let y = nn.softshrink(x);
    const epsilon = 1e-4;
    let expectedY = mx.array([0.5, -0.5, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);

    y = nn.softshrink(x, 0.7);
    expectedY = mx.array([0.3, -0.3, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('celu', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    let y = nn.celu(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([1.0, -0.6321, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);

    y = new nn.CELU(1.1).forward(x);
    const expectedYSecond = mx.array([1.0, -0.6568, 0.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedYSecond)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('logSoftmax', () => {
    const x = mx.array([1.0, 2.0, 3.0]);
    const y = nn.logSoftmax(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([-2.4076, -1.4076, -0.4076]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('logSigmoid', () => {
    const x = mx.array([1.0, -1.0, 0.0]);
    const y = nn.logSigmoid(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([-0.3133, -1.3133, -0.6931]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [3]);
    assert.equal(y.dtype, mx.float32);
  });

  it('prelu', () => {
    assertArrayAllTrue(mx.arrayEqual(new nn.PReLU().forward(mx.array([1.0, -1.0, 0.0, 0.5])),
      mx.array([1.0, -0.25, 0.0, 0.5])));
  });

  it('mish', () => {
    assertArrayAllTrue(mx.isclose(
      new nn.Mish().forward(mx.array([1.0, -1.0, 0.0, 0.5])),
      mx.array([0.8651, -0.3034, 0.0000, 0.3752]),
      1e-3
    ));
  });

  it('hardswish', () => {
    const x = mx.array([-3.0, -1.5, 0.0, 1.5, 3.0]);
    const y = nn.hardswish(x);
    const epsilon = 1e-4;
    const expectedY = mx.array([0.0, -0.375, 0.0, 1.125, 3.0]);
    assertArrayAllTrue(mx.less(mx.abs(mx.subtract(y, expectedY)), epsilon));
    assert.deepEqual(y.shape, [5]);
    assert.equal(y.dtype, mx.float32);
  });

  it('glu', () => {
    const x = mx.array([[[1.0, 2.0, 3.0, 4.0]]], mx.float32);
    const y = mx.array([[[0.952574, 1.96403]]], mx.float32);
    const out = nn.glu(x);
    assertArrayAllTrue(mx.isclose(out, y));
  });

  it('hardTanh', () => {
    const x = mx.array([1.0, -2.0, 0.0, 0.5, 2.0]);
    const y = nn.hardTanh(x);
    const expectedY = mx.array([1.0, -1.0, 0.0, 0.5, 1.0]);
    assertArrayAllTrue(mx.arrayEqual(y, expectedY));
    assert.deepEqual(y.shape, [5]);
    assert.equal(y.dtype, mx.float32);
  });

  it('hardShrink', () => {
    let x = mx.array([1.0, -0.5, 0.0, 0.5, -1.5]);
    let y = nn.hardShrink(x);
    let expectedY = mx.array([1.0, 0.0, 0.0, 0.0, -1.5]);
    assertArrayAllTrue(mx.arrayEqual(y, expectedY));
    assert.deepEqual(y.shape, [5]);
    assert.equal(y.dtype, mx.float32);

    y = nn.hardShrink(x, 0.1);
    expectedY = mx.array([1.0, -0.5, 0.0, 0.5, -1.5]);
    assertArrayAllTrue(mx.arrayEqual(y, expectedY));
    assert.deepEqual(y.shape, [5]);
    assert.equal(y.dtype, mx.float32);
  });

  it('rope', () => {
    [[], [false], [undefined, 10000], [undefined, undefined, 0.25]].forEach((args: any) => {
      const rope = new nn.RoPE(4, ...args);
      const shape = [1, 3, 4];
      const x = mx.random.uniform(0, 1, shape);
      let y = rope.forward(x);
      assert.deepEqual(y.shape, shape);
      assert.equal(y.dtype, mx.float32);

      y = rope.forward(x, 3);
      assert.deepEqual(y.shape, shape);

      y = rope.forward(x.astype(mx.float16));
      assert.equal(y.dtype, mx.float16);
    });
  });

  it('alibi', () => {
    const alibi = new nn.ALiBi();
    const shape = [1, 8, 20, 20];
    const x = mx.random.uniform(0, 1, shape);
    let y = alibi.forward(x);
    assert.deepEqual(y.shape, shape);
    assert.equal(y.dtype, mx.float32);

    y = alibi.forward(x.astype(mx.float16));
    assert.equal(y.dtype, mx.float16);
  });

  it('dropout', () => {
    let x = mx.ones([2, 4]);
    let y = new nn.Dropout(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.float32);

    x = mx.ones([2, 4], mx.bfloat16);
    y = new nn.Dropout(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.bfloat16);

    x = mx.ones([2, 4], mx.float16);
    y = new nn.Dropout(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.float16);
  });

  it('dropout2d', () => {
    let x = mx.ones([2, 4, 4, 4]);
    let y = new nn.Dropout2d(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.float32);

    x = mx.ones([2, 4, 4, 4], mx.bfloat16);
    y = new nn.Dropout2d(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.bfloat16);

    x = mx.ones([2, 4, 4, 4], mx.float16);
    y = new nn.Dropout2d(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.float16);
  });

  it('dropout3d', () => {
    let x = mx.ones([2, 4, 4, 4, 4]);
    let y = new nn.Dropout3d(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.float32);

    x = mx.ones([2, 4, 4, 4, 4], mx.bfloat16);
    y = new nn.Dropout3d(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.bfloat16);

    x = mx.ones([2, 4, 4, 4, 4], mx.float16);
    y = new nn.Dropout3d(0.5).forward(x);
    assert.deepEqual(y.shape, x.shape);
    assert.equal(y.dtype, mx.float16);
  });

  it('upsample', () => {
    const b = 1, h = 2, w = 2, c = 1;
    const scaleFactor = 2;
    const upsampleNearest = new nn.Upsample(scaleFactor, 'nearest', true);
    const upsampleBilinear = new nn.Upsample(scaleFactor, 'linear', true);
    const upsampleNearestNoAlignCorners = new nn.Upsample(scaleFactor, 'nearest', false);
    const upsampleBilinearNoAlignCorners = new nn.Upsample(scaleFactor, 'linear', false);

    // Test single feature map, align corners
    let x = mx.arange(b * h * w * c).reshape([b, c, h, w]).transpose([0, 2, 3, 1]);
    const expectedNearest = mx.array(
      [[[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]]]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleNearest.forward(x),
                                  expectedNearest));

    const expectedBilinear = mx.array(
      [
        [
          [
            [0, 0.333333, 0.666667, 1],
            [0.666667, 1, 1.33333, 1.66667],
            [1.33333, 1.66667, 2, 2.33333],
            [2, 2.33333, 2.66667, 3],
          ]
        ]
      ]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleBilinear.forward(x),
                                  expectedBilinear));

    // Test single feature map, no align corners
    x = mx.arange(b * h * w * c).reshape([b, c, h, w]).transpose([0, 2, 3, 1]);
    const expectedNearestNoAlignCorners = mx.array(
      [[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleNearestNoAlignCorners.forward(x),
                                  expectedNearestNoAlignCorners));

    const expectedBilinearNoAlignCorners = mx.array(
      [
        [
          [
            [1.0000, 1.2500, 1.7500, 2.0000],
            [1.5000, 1.7500, 2.2500, 2.5000],
            [2.5000, 2.7500, 3.2500, 3.5000],
            [3.0000, 3.2500, 3.7500, 4.0000],
          ]
        ]
      ]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleBilinearNoAlignCorners.forward(x),
                                  expectedBilinearNoAlignCorners));

    // Test a more complex batch
    const b1 = 2, h1 = 3, w1 = 3, c1 = 2;
    const scaleFactor1 = 2;
    let x1 = mx.arange(b1 * h1 * w1 * c1).reshape([b1, c1, h1, w1]).transpose([0, 2, 3, 1]);
    const upsampleNearest1 = new nn.Upsample(scaleFactor1, 'nearest', true);
    const upsampleBilinear1 = new nn.Upsample(scaleFactor1, 'linear', true);

    const expectedNearest1 = mx.array(
      [
        [
          [
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
            [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
          ],
          [
            [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
            [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
            [12.0, 12.0, 13.0, 13.0, 14.0, 14.0],
            [12.0, 12.0, 13.0, 13.0, 14.0, 14.0],
            [15.0, 15.0, 16.0, 16.0, 17.0, 17.0],
            [15.0, 15.0, 16.0, 16.0, 17.0, 17.0],
          ],
        ],
        [
          [
            [18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
            [18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
            [21.0, 21.0, 22.0, 22.0, 23.0, 23.0],
            [21.0, 21.0, 22.0, 22.0, 23.0, 23.0],
            [24.0, 24.0, 25.0, 25.0, 26.0, 26.0],
            [24.0, 24.0, 25.0, 25.0, 26.0, 26.0],
          ],
          [
            [27.0, 27.0, 28.0, 28.0, 29.0, 29.0],
            [27.0, 27.0, 28.0, 28.0, 29.0, 29.0],
            [30.0, 30.0, 31.0, 31.0, 32.0, 32.0],
            [30.0, 30.0, 31.0, 31.0, 32.0, 32.0],
            [33.0, 33.0, 34.0, 34.0, 35.0, 35.0],
            [33.0, 33.0, 34.0, 34.0, 35.0, 35.0],
          ],
        ],
      ]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleNearest1.forward(x1),
                                  expectedNearest1));

    const expectedBilinear1 = mx.array(
      [
        [
          [
            [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            [1.2, 1.6, 2.0, 2.4, 2.8, 3.2],
            [2.4, 2.8, 3.2, 3.6, 4.0, 4.4],
            [3.6, 4.0, 4.4, 4.8, 5.2, 5.6],
            [4.8, 5.2, 5.6, 6.0, 6.4, 6.8],
            [6.0, 6.4, 6.8, 7.2, 7.6, 8.0],
          ],
          [
            [9.0, 9.4, 9.8, 10.2, 10.6, 11.0],
            [10.2, 10.6, 11.0, 11.4, 11.8, 12.2],
            [11.4, 11.8, 12.2, 12.6, 13.0, 13.4],
            [12.6, 13.0, 13.4, 13.8, 14.2, 14.6],
            [13.8, 14.2, 14.6, 15.0, 15.4, 15.8],
            [15.0, 15.4, 15.8, 16.2, 16.6, 17.0],
          ],
        ],
        [
          [
            [18.0, 18.4, 18.8, 19.2, 19.6, 20.0],
            [19.2, 19.6, 20.0, 20.4, 20.8, 21.2],
            [20.4, 20.8, 21.2, 21.6, 22.0, 22.4],
            [21.6, 22.0, 22.4, 22.8, 23.2, 23.6],
            [22.8, 23.2, 23.6, 24.0, 24.4, 24.8],
            [24.0, 24.4, 24.8, 25.2, 25.6, 26.0],
          ],
          [
            [27.0, 27.4, 27.8, 28.2, 28.6, 29.0],
            [28.2, 28.6, 29.0, 29.4, 29.8, 30.2],
            [29.4, 29.8, 30.2, 30.6, 31.0, 31.4],
            [30.6, 31.0, 31.4, 31.8, 32.2, 32.6],
            [31.8, 32.2, 32.6, 33.0, 33.4, 33.8],
            [33.0, 33.4, 33.8, 34.2, 34.6, 35.0],
          ],
        ],
      ]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleBilinear1.forward(x1),
                                  expectedBilinear1));

    // Test different height and width scaleFactor
    const b2 = 1, h2 = 2, w2 = 2, c2 = 2;
    const x2 = mx.arange(b2 * h2 * w2 * c2).reshape([b2, c2, h2, w2]).transpose([0, 2, 3, 1]);
    const upsampleNearest2 = new nn.Upsample([2, 3], 'nearest', true);
    const upsampleBilinear2 = new nn.Upsample([2, 3], 'linear', true);

    const expectedNearest2 = mx.array(
      [
        [
          [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [2, 2, 2, 3, 3, 3],
            [2, 2, 2, 3, 3, 3],
          ],
          [
            [4, 4, 4, 5, 5, 5],
            [4, 4, 4, 5, 5, 5],
            [6, 6, 6, 7, 7, 7],
            [6, 6, 6, 7, 7, 7],
          ],
        ]
      ]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleNearest2.forward(x2),
                                  expectedNearest2));

    const expectedBilinear2 = mx.array(
      [
        [
          [
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            [0.666667, 0.866667, 1.06667, 1.26667, 1.46667, 1.66667],
            [1.33333, 1.53333, 1.73333, 1.93333, 2.13333, 2.33333],
            [2, 2.2, 2.4, 2.6, 2.8, 3],
          ],
          [
            [4, 4.2, 4.4, 4.6, 4.8, 5],
            [4.66667, 4.86667, 5.06667, 5.26667, 5.46667, 5.66667],
            [5.33333, 5.53333, 5.73333, 5.93333, 6.13333, 6.33333],
            [6, 6.2, 6.4, 6.6, 6.8, 7],
          ],
        ]
      ]
    ).transpose([0, 2, 3, 1]);
    assertArrayAllTrue(mx.isclose(upsampleBilinear2.forward(x2),
                                  expectedBilinear2));

    assert.equal(
      new nn.Upsample(2).toString(),
      "Upsample(scaleFactor=2, mode='nearest', alignCorners=false)"
    );
    assert.equal(
      new nn.Upsample([2, 3]).toString(),
      "Upsample(scaleFactor=2,3, mode='nearest', alignCorners=false)"
    );
  });

  describe('pooling', () => {
    it('1d', () => {
      const x = mx.array(
        [
          [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
          [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
        ]
      );
      const expectedMaxPoolOutputNoPaddingStride1 = [
        [[3, 4, 5], [6, 7, 8], [9, 10, 11]],
        [[15, 16, 17], [18, 19, 20], [21, 22, 23]],
      ];
      const expectedMaxPoolOutputNoPaddingStride2 = [
        [[3, 4, 5], [9, 10, 11]],
        [[15, 16, 17], [21, 22, 23]],
      ];
      const expectedMaxPoolOutputPadding1Stride2 = [
        [[0, 1, 2], [6, 7, 8], [9, 10, 11]],
        [[12, 13, 14], [18, 19, 20], [21, 22, 23]],
      ];
      const expectedMaxPoolOutputPadding1Stride2Kernel3 = [
        [[3, 4, 5], [9, 10, 11]],
        [[15, 16, 17], [21, 22, 23]],
      ];
      const expectedAvgPoolOutputNoPaddingStride1 = [
        [
          [1.5000, 2.5000, 3.5000],
          [4.5000, 5.5000, 6.5000],
          [7.5000, 8.5000, 9.5000],
        ],
        [
          [13.5000, 14.5000, 15.5000],
          [16.5000, 17.5000, 18.5000],
          [19.5000, 20.5000, 21.5000],
        ],
      ];
      const expectedAvgPoolOutputNoPaddingStride2 = [
        [[1.5000, 2.5000, 3.5000], [7.5000, 8.5000, 9.5000]],
        [[13.5000, 14.5000, 15.5000], [19.5000, 20.5000, 21.5000]],
      ];
      const expectedAvgPoolOutputPadding1Stride2 = [
        [
          [0.0000, 0.5000, 1.0000],
          [4.5000, 5.5000, 6.5000],
          [4.5000, 5.0000, 5.5000],
        ],
        [
          [6.0000, 6.5000, 7.0000],
          [16.5000, 17.5000, 18.5000],
          [10.5000, 11.0000, 11.5000],
        ],
      ];
      const expectedAvgPoolOutputPadding1Kernel3 = [
        [[1, 1.66667, 2.33333], [6, 7, 8]],
        [[9, 9.66667, 10.3333], [18, 19, 20]],
      ];
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool1d(2, 1, 0).forward(x),
        expectedMaxPoolOutputNoPaddingStride1,
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool1d(2, 2, 0).forward(x),
        expectedMaxPoolOutputNoPaddingStride2,
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool1d(2, 2, 1).forward(x),
        expectedMaxPoolOutputPadding1Stride2,
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool1d(3, 2, 1).forward(x),
        expectedMaxPoolOutputPadding1Stride2Kernel3,
      ));
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool1d(2, 1, 0).forward(x),
        expectedAvgPoolOutputNoPaddingStride1,
      ));
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool1d(2, 2, 0).forward(x),
        expectedAvgPoolOutputNoPaddingStride2,
      ));
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool1d(2, 2, 1).forward(x),
        expectedAvgPoolOutputPadding1Stride2,
      ));
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool1d(3, 2, 1).forward(x),
        expectedAvgPoolOutputPadding1Kernel3,
      ));
    });

    it('2d', () => {
      const x = mx.array(
        [
          [
            [[0, 16], [1, 17], [2, 18], [3, 19]],
            [[4, 20], [5, 21], [6, 22], [7, 23]],
            [[8, 24], [9, 25], [10, 26], [11, 27]],
            [[12, 28], [13, 29], [14, 30], [15, 31]],
          ]
        ]
      );
      const expectedMaxPoolOutputNoPaddingStride1 = [
        [
          [[5, 21], [6, 22], [7, 23]],
          [[9, 25], [10, 26], [11, 27]],
          [[13, 29], [14, 30], [15, 31]],
        ],
      ];
      const expectedMaxPoolOutputNoPaddingStride2 = [
        [[[5, 21], [7, 23]], [[13, 29], [15, 31]]]
      ];
      const expectedMaxPoolOutputPadding1 = [
        [
          [[0, 16], [2, 18], [3, 19]],
          [[8, 24], [10, 26], [11, 27]],
          [[12, 28], [14, 30], [15, 31]]
        ]
      ];
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool2d(2, 1, 0).forward(x),
        expectedMaxPoolOutputNoPaddingStride1
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool2d(2, 2, 0).forward(x),
        expectedMaxPoolOutputNoPaddingStride2
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool2d(2, 2, 1).forward(x),
        expectedMaxPoolOutputPadding1
      ));
    });

    it('average', () => {
      const x = mx.array(
        [
          [
            [[0, 16], [1, 17], [2, 18], [3, 19]],
            [[4, 20], [5, 21], [6, 22], [7, 23]],
            [[8, 24], [9, 25], [10, 26], [11, 27]],
            [[12, 28], [13, 29], [14, 30], [15, 31]],
          ]
        ]
      );
      const expectedMeanPoolOutputNoPaddingStride1 = [
        [
          [[2.5000, 18.5000], [3.5000, 19.5000], [4.5000, 20.5000]],
          [[6.5000, 22.5000], [7.5000, 23.5000], [8.5000, 24.5000]],
          [[10.5000, 26.5000], [11.5000, 27.5000], [12.5000, 28.5000]],
        ],
      ];
      const expectedMeanPoolOutputNoPaddingStride2 = [
        [
          [[2.5000, 18.5000], [4.5000, 20.5000]],
          [[10.5000, 26.5000], [12.5000, 28.5000]],
        ],
      ];
      const expectedMeanPoolOutputPadding1 = [
        [
          [[0.0000, 4.0000], [0.7500, 8.7500], [0.7500, 4.7500]],
          [[3.0000, 11.0000], [7.5000, 23.5000], [4.5000, 12.5000]],
          [[3.0000, 7.0000], [6.7500, 14.7500], [3.7500, 7.7500]],
        ],
      ];
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool2d(2, 1, 0).forward(x),
        expectedMeanPoolOutputNoPaddingStride1,
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.AvgPool2d(2, 2, 0).forward(x),
        expectedMeanPoolOutputNoPaddingStride2,
      ));
      assertArrayAllTrue(mx.arrayEqual(
        new nn.AvgPool2d(2, 2, 1).forward(x),
        expectedMeanPoolOutputPadding1,
      ));
    });

    it('multiBatch', () => {
      const x = mx.array(
        [
          [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13], [14, 15]],
            [[16, 17], [18, 19], [20, 21], [22, 23]],
            [[24, 25], [26, 27], [28, 29], [30, 31]],
          ],
          [
            [[32, 33], [34, 35], [36, 37], [38, 39]],
            [[40, 41], [42, 43], [44, 45], [46, 47]],
            [[48, 49], [50, 51], [52, 53], [54, 55]],
            [[56, 57], [58, 59], [60, 61], [62, 63]],
          ],
        ]
      );
      const expectedMaxPoolOutput = [
        [[[10, 11], [14, 15]], [[26, 27], [30, 31]]],
        [[[42, 43], [46, 47]], [[58, 59], [62, 63]]],
      ];
      const expectedAvgPoolOutput = [
        [[[2.22222, 2.66667], [5.33333, 6]], [[11.3333, 12], [20, 21]]],
        [[[16.4444, 16.8889], [26.6667, 27.3333]], [[32.6667, 33.3333], [52, 53]]],
      ];
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool2d(3, 2, 1).forward(x),
        expectedMaxPoolOutput,
      ));
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool2d(3, 2, 1).forward(x),
        expectedAvgPoolOutput,
      ));
    });

    it('irregular', () => {
      const x = mx.array([
        [
          [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
          [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
          [[24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35]],
          [[36, 37, 38], [39, 40, 41], [42, 43, 44], [45, 46, 47]],
        ],
        [
          [[48, 49, 50], [51, 52, 53], [54, 55, 56], [57, 58, 59]],
          [[60, 61, 62], [63, 64, 65], [66, 67, 68], [69, 70, 71]],
          [[72, 73, 74], [75, 76, 77], [78, 79, 80], [81, 82, 83]],
          [[84, 85, 86], [87, 88, 89], [90, 91, 92], [93, 94, 95]],
        ]
      ]);
      const expectedIrregularMaxPoolOutput = [
        [
          [[3, 4, 5], [6, 7, 8], [9, 10, 11], [9, 10, 11], [9, 10, 11]],
          [[39, 40, 41], [42, 43, 44], [45, 46, 47], [45, 46, 47], [45, 46, 47]],
        ],
        [
          [[51, 52, 53], [54, 55, 56], [57, 58, 59], [57, 58, 59], [57, 58, 59]],
          [[87, 88, 89], [90, 91, 92], [93, 94, 95], [93, 94, 95], [93, 94, 95]],
        ]
      ];
      const expectedIrregularAveragePoolOutput = [
        [
          [[0.3750, 0.6250, 0.8750], [1.1250, 1.5000, 1.8750], [2.2500, 2.7500, 3.2500], [2.2500, 2.6250, 3.0000], [1.8750, 2.1250, 2.3750]],
          [[15.7500, 16.2500, 16.7500], [24.7500, 25.5000, 26.2500], [34.5000, 35.5000, 36.5000], [27.0000, 27.7500, 28.5000], [18.7500, 19.2500, 19.7500]],
        ],
        [
          [[12.3750, 12.6250, 12.8750], [19.1250, 19.5000, 19.8750], [26.2500, 26.7500, 27.2500], [20.2500, 20.6250, 21.0000], [13.8750, 14.1250, 14.3750]],
          [[39.7500, 40.2500, 40.7500], [60.7500, 61.5000, 62.2500], [82.5000, 83.5000, 84.5000], [63.0000, 63.7500, 64.5000], [42.7500, 43.2500, 43.7500]],
        ]
      ];
      assertArrayAllTrue(mx.arrayEqual(
        new nn.MaxPool2d([2, 4], [3, 1], [1, 2]).forward(x),
        mx.array(expectedIrregularMaxPoolOutput),
      ));
      assertArrayAllTrue(mx.allclose(
        new nn.AvgPool2d([2, 4], [3, 1], [1, 2]).forward(x),
        mx.array(expectedIrregularAveragePoolOutput),
      ));
    });

    it('toString', () => {
      assert.equal(
        new nn.MaxPool1d(3, undefined, 2).toString(),
        'MaxPool1d(kernelSize=3, stride=3, padding=2)',
      );
      assert.equal(
        new nn.AvgPool1d(2, 3).toString(),
        'AvgPool1d(kernelSize=2, stride=3, padding=0)',
      );
      assert.equal(
        new nn.MaxPool2d(3, 2, 1).toString(),
        'MaxPool2d(kernelSize=3,3, stride=2,2, padding=1,1)',
      );
      assert.equal(
        new nn.AvgPool2d([1, 2], 2, [1, 2]).toString(),
        'AvgPool2d(kernelSize=1,2, stride=2,2, padding=1,2)',
      );
    });

    it('3dPooling', () => {
      const x = mx.array([
        [
          [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
          ],
          [
            [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
            [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
            [[45, 46, 47], [48, 49, 50], [51, 52, 53]],
          ],
        ],
      ]);
      const expectedMaxPoolOutputNoPaddingStride1 = [[[[[39, 40, 41], [42, 43, 44]], [[48, 49, 50], [51, 52, 53]]]]];
      const expectedMaxPoolOutputNoPaddingStride2 = [[[[[39, 40, 41]]]]];
      const expectedMaxPoolOutputPadding1 = [
        [
          [[[0, 1, 2], [6, 7, 8]], [[18, 19, 20], [24, 25, 26]]],
          [[[27, 28, 29], [33, 34, 35]], [[45, 46, 47], [51, 52, 53]]],
        ],
      ];
      const expectedIrregularMaxPoolOutput = [
        [
          [[[9, 10, 11], [12, 13, 14], [15, 16, 17]]],
          [[[36, 37, 38], [39, 40, 41], [42, 43, 44]]],
        ],
      ];
      assert.deepEqual(
        new nn.MaxPool3d(2, 1, 0).forward(x).tolist(),
        expectedMaxPoolOutputNoPaddingStride1
      );
      assert.deepEqual(
        new nn.MaxPool3d(2, 2, 0).forward(x).tolist(),
        expectedMaxPoolOutputNoPaddingStride2
      );
      assert.deepEqual(
        new nn.MaxPool3d(2, 2, 1).forward(x).tolist(),
        expectedMaxPoolOutputPadding1
      );
      assert.deepEqual(
        new nn.MaxPool3d([1, 2, 1], [1, 2, 1]).forward(x).tolist(),
        expectedIrregularMaxPoolOutput
      );
      assert.deepEqual(
        new nn.MaxPool3d(3, 3, 2).toString(),
        "MaxPool3d(kernelSize=3,3,3, stride=3,3,3, padding=2,2,2)"
      );

      const expectedAvgPoolOutputNoPaddingStride1 = [
        [[[[19.5, 20.5, 21.5], [22.5, 23.5, 24.5]], [[28.5, 29.5, 30.5], [31.5, 32.5, 33.5]]]]
      ];
      const expectedAvgPoolOutputNoPaddingStride2 = [[[[[19.5, 20.5, 21.5]]]]];
      const expectedAvgPoolOutputPadding1 = [
        [
          [[[0, 0.125, 0.25], [1.125, 1.375, 1.625]], [[3.375, 3.625, 3.875], [9, 9.5, 10]]],
          [[[3.375, 3.5, 3.625], [7.875, 8.125, 8.375]], [[10.125, 10.375, 10.625], [22.5, 23, 23.5]]],
        ],
      ];
      const expectedIrregularAvgPoolOutput = [
        [
          [[[4.5, 5.5, 6.5], [7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]],
          [[[31.5, 32.5, 33.5], [34.5, 35.5, 36.5], [37.5, 38.5, 39.5]]],
        ],
      ];
      assert.deepEqual(
        new nn.AvgPool3d(2, 1, 0).forward(x).tolist(),
        expectedAvgPoolOutputNoPaddingStride1
      );
      assert.deepEqual(
        new nn.AvgPool3d(2, 2, 0).forward(x).tolist(),
        expectedAvgPoolOutputNoPaddingStride2
      );
      assert.deepEqual(
        new nn.AvgPool3d(2, 2, 1).forward(x).tolist(),
        expectedAvgPoolOutputPadding1
      );
      assert.deepEqual(
        new nn.AvgPool3d([1, 2, 1], [1, 2, 1]).forward(x).tolist(),
        expectedIrregularAvgPoolOutput
      );
      assert.deepEqual(
        new nn.AvgPool3d(3, 3, 2).toString(),
        "AvgPool3d(kernelSize=3,3,3, stride=3,3,3, padding=2,2,2)"
      );
    });  });

  it('setDtype', () => {
    const assertDtype = (layer: nn.Linear, dtype: mx.Dtype) => {
      const parameters = layer.parameters();
      for (const key of Object.keys(parameters)) {
        assert.equal((parameters[key] as mx.array).dtype, dtype, `dtype mismatch for ${key}`);
      }
    };

    const layer = new nn.Linear(4, 8, true);
    assertDtype(layer, mx.float32);

    layer.setDtype(mx.bfloat16);
    assertDtype(layer, mx.bfloat16);

    layer.setDtype(mx.float32, () => false);
    assertDtype(layer, mx.bfloat16);

    layer.setDtype(mx.int32, () => true);
    assertDtype(layer, mx.int32);

    layer.setDtype(mx.int64, null);
    assertDtype(layer, mx.int64);

    layer.setDtype(mx.int16, x => mx.issubdtype(x, mx.integer));
    assertDtype(layer, mx.int16);
  });

  it('rnn', () => {
    let layer = new nn.RNN(5, 12, true);
    let inp = mx.random.normal([2, 25, 5]);

    let hOut = layer.forward(inp);
    assert.deepEqual(hOut.shape, [2, 25, 12]);

    layer = new nn.RNN(5, 12, false, x => mx.maximum(0, x));

    hOut = layer.forward(inp);
    assert.deepEqual(hOut.shape, [2, 25, 12]);

    assert.throws(() => new nn.RNN(5, 12, undefined, 'tanh' as any));

    inp = mx.random.normal([44, 5]);
    hOut = layer.forward(inp);
    assert.deepEqual(hOut.shape, [44, 12]);

    hOut = layer.forward(inp, hOut.index(-1));
    assert.deepEqual(hOut.shape, [44, 12]);
  });

  it('gru', () => {
    let layer = new nn.GRU(5, 12, true);
    let inp = mx.random.normal([2, 25, 5]);

    let hOut = layer.forward(inp);
    assert.deepEqual(hOut.shape, [2, 25, 12]);

    hOut = layer.forward(inp, hOut.index(mx.Slice(), -1, mx.Slice()));
    assert.deepEqual(hOut.shape, [2, 25, 12]);

    inp = mx.random.normal([44, 5]);
    hOut = layer.forward(inp);
    assert.deepEqual(hOut.shape, [44, 12]);

    hOut = layer.forward(inp, hOut.index(-1, mx.Slice()));
    assert.deepEqual(hOut.shape, [44, 12]);
  });

  it('lstm', () => {
    let layer = new nn.LSTM(5, 12);
    let inp = mx.random.normal([2, 25, 5]);

    let [hOut, cOut] = layer.forward(inp);
    assert.deepEqual(hOut.shape, [2, 25, 12]);
    assert.deepEqual(cOut.shape, [2, 25, 12]);

    [hOut, cOut] = layer.forward(inp,
                                 hOut.index(mx.Slice(), -1, mx.Slice()),
                                 cOut.index(mx.Slice(), -1, mx.Slice()));
    assert.deepEqual(hOut.shape, [2, 25, 12]);
    assert.deepEqual(cOut.shape, [2, 25, 12]);

    inp = mx.random.normal([44, 5]);
    [hOut, cOut] = layer.forward(inp);
    assert.deepEqual(hOut.shape, [44, 12]);
    assert.deepEqual(cOut.shape, [44, 12]);

    inp = mx.random.normal([44, 5]);
    [hOut, cOut] = layer.forward(inp,
                                 hOut.index(-1, mx.Slice()),
                                 cOut.index(-1, mx.Slice()));
    assert.deepEqual(hOut.shape, [44, 12]);
    assert.deepEqual(cOut.shape, [44, 12]);
  });

  it('quantizedEmbedding', () => {
    const emb = new nn.Embedding(32, 256);
    const qemb = nn.QuantizedEmbedding.fromEmbedding(emb, undefined, 8);
    let x = mx.array([2, 6, 9, 3, 0, 3], mx.int32);
    let y = emb.forward(x);
    let yq = qemb.forward(x);
    assert.isBelow(mx.abs(mx.subtract(y, yq)).max().item() as number,
                   qemb.scales.max().item() as number);

    x = mx.random.uniform(0, 1, [2, 256]);
    y = emb.asLinear(x);
    yq = qemb.asLinear(x);

    const cosine = (a, b) => {
      const ab = mx.multiply(a, b).sum(-1);
      const aa = mx.linalg.norm(a, -1);
      const bb = mx.linalg.norm(b, -1);
      return mx.divide(mx.divide(ab, aa), bb);
    };

    assert.isAbove(cosine(y, yq).min().item() as number, 0.99);
  });

  it('causalMask', () => {
    let mask = nn.MultiHeadAttention.createAdditiveCausalMask(4, mx.float16);
    assert.isFalse(mx.any(mx.isnan(mask)).item());
    assert.isBelow(mask.index(0, -1).item() as number, 0);

    mask = nn.MultiHeadAttention.createAdditiveCausalMask(4, mx.bfloat16);
    assert.isFalse(mx.any(mx.isnan(mask)).item());
    assert.isBelow(mask.index(0, -1).item() as number, 0);
  });

  it('attention', () => {
    const attn = new nn.MultiHeadAttention(32, 4);
    const x = mx.random.normal([2, 5, 32]);
    const out = attn.forward(x, x, x);
    assert.deepEqual(out.shape, x.shape);
  });
});
