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
});
