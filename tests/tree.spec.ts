import mlx from '..';
import {assert} from 'chai';

describe('treeUtils', () => {
  it('treeMap', () => {
    const tree = {a: 0, b: 1, c: 2};
    const result = mlx.utils.treeMap((x: number) => x + 1, tree);
    assert.deepEqual(result, {a: 1, b: 2, c: 3});
  });

  it('treeFlatten', () => {
    const tree = [{a: 1, b: 2}, 'c'];
    const values = [1, 2, 'c'];
    const flatTree = mlx.utils.treeFlatten(tree);

    assert.deepEqual(flatTree.map(subTree => subTree[1]), values);
    assert.deepEqual(mlx.utils.treeUnflatten(flatTree), tree);
  });
});
