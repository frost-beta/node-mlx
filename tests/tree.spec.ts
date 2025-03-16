import {core as mx, nn, utils} from '..';
import {assert} from 'chai';

describe('treeUtils', () => {
  it('treeMap', () => {
    const tree = {a: 0, b: 1, c: 2};
    const result = utils.treeMap((x: number) => x + 1, tree);
    assert.deepEqual(result, {a: 1, b: 2, c: 3});
  });

  it('treeFlatten', () => {
    const tree = [{a: 1, b: 2}, 'c'];
    const values = [1, 2, 'c'];
    const flatTree = utils.treeFlatten(tree);

    assert.deepEqual(flatTree.map(subTree => subTree[1]), values);
    assert.deepEqual(utils.treeUnflatten(flatTree), tree);
  });

  it('merge', () => {
    const t1 = {'a': 0};
    const t2 = {'b': 1};
    let t = utils.treeMerge(t1, t2);
    assert.deepEqual({'a': 0, 'b': 1}, t);
    assert.throws(() => utils.treeMerge(t1, t1), Error);
    assert.throws(() => utils.treeMerge(t, t1), Error);

    const mod1 = new nn.Sequential(new nn.Linear(2, 2), new nn.Linear(2, 2));
    const mod2 = new nn.Sequential(new nn.Linear(2, 2), new nn.Linear(2, 2));
    const mod = new nn.Sequential(mod1, mod2);

    const params1 = {'layers': [mod1.parameters()]};
    const params2 = {'layers': [null, mod2.parameters()]};
    const params = utils.treeMerge(params1, params2);
    const flattened1 = utils.treeFlatten(params);
    const flattened2 = utils.treeFlatten(mod.parameters());
    for (let i = 0; i < flattened1.length; i++) {
      assert.deepEqual(flattened1[i][0], flattened2[i][0]);
      assert.isTrue(mx.arrayEqual(flattened1[i][1] as any,
                                  flattened2[i][1] as any).item());
    }
  });
});
