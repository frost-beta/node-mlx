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
    assert.isTrue(leaves[0][1] === (m.layers[0] as nn.Module).layers[0]);
    assert.isTrue(leaves[1][1] === (m.layers[1] as nn.Module).layers[0]);
    assert.isTrue(leaves[2][1] === (m.layers[1] as nn.Module).layers[1]);
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
      val?: mx.array;

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
    assert.equal(model.val.item(), mx.array(1.0).item());
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
});

describe('layers', () => {
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

    g.weight = mx.multiply(g.weight, 2);
    g.bias = mx.add(g.bias, 3);
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

    g.weight = mx.multiply(g.weight, 2);
    g.bias = mx.add(g.bias, 3);
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
    assertArrayAllTrue(mx.equal(bn.runningMean, mx.zerosLike(bn.runningMean)));
    assertArrayAllTrue(mx.equal(bn.runningVar, mx.onesLike(bn.runningVar)));
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
    assertArrayAllTrue(mx.allclose(bn.runningMean, expectedMean, undefined, 1e-5));
    assertArrayAllTrue(mx.allclose(bn.runningVar, expectedVar, undefined, 1e-5));

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
    assertArrayAllTrue(mx.equal(bn.runningMean, mx.zerosLike(bn.runningMean)));
    assertArrayAllTrue(mx.equal(bn.runningVar, mx.onesLike(bn.runningVar)));
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
    assertArrayAllTrue(mx.allclose(bn.runningMean, expectedMean, undefined, 1e-5));
    assertArrayAllTrue(mx.allclose(bn.runningVar, expectedVar, undefined, 1e-5));

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
    let runningMean = batchNorm.runningMean;
    let runningVar = batchNorm.runningVar;

    let data = mx.random.normal([batchSize, numFeatures]);
    let normalizedData = batchNorm.forward(data);
    let means = data.mean(0);
    let variances = data.variance(0);
    runningMean = mx.add(mx.multiply(runningMean, (1 - momentum)), mx.multiply(means, momentum));
    runningVar = mx.add(mx.multiply(runningVar, (1 - momentum)), mx.multiply(variances, momentum));
    assertArrayAllTrue(mx.allclose(batchNorm.runningMean, runningMean, 1e-5));
    assertArrayAllTrue(mx.allclose(batchNorm.runningVar, runningVar, 1e-5));

    batchNorm = new nn.BatchNorm(numFeatures);
    batchNorm.train();
    runningMean = batchNorm.runningMean;
    runningVar = batchNorm.runningVar;
    data = mx.random.normal([batchSize, h, w, numFeatures]);

    normalizedData = batchNorm.forward(data);
    means = data.mean([0, 1, 2]);
    variances = data.variance([0, 1, 2]);
    runningMean = mx.add(mx.multiply(runningMean, (1 - momentum)), mx.multiply(means, momentum));
    runningVar = mx.add(mx.multiply(runningVar, (1 - momentum)), mx.multiply(variances, momentum));
    assertArrayAllTrue(mx.allclose(batchNorm.runningMean, runningMean, 1e-5));
    assertArrayAllTrue(mx.allclose(batchNorm.runningVar, runningVar, 1e-5));

    assert.deepEqual(batchNorm.runningMean.shape, runningMean.shape);
    assert.deepEqual(batchNorm.runningVar.shape, runningVar.shape);
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

    c = new nn.Conv1d(CIn, COut, ks, undefined, undefined, undefined, false);
    assert.notProperty(c.parameters(), 'bias');
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
  });

  beforeEach(function() {
    // FIXME(zcbenz): Compilation fails on QEMU in CI.
    if (process.platform == 'linux' && process.arch == 'arm64')
      this.skip();
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
    this.timeout(10 * 1000);

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
    const y2 = nn.activations.sigmoid(x);
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
});
