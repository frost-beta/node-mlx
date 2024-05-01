/**
 * Applies `fn` to the leaves of the Object `tree` and returns a new Array with
 * the results.
 *
 * @remarks
 *
 * If `rest` is provided, every item is assumed to be a superset of `tree` and
 * the corresponding leaves are provided as extra positional arguments to `fn`.
 *
 * The argument `isLeaf` decides what constitutes a leaf from `tree` similar to
 * `treeFlatten`.
 *
 * Example:
 *
 * ```ts
 * import {core as mx, nn, utils} from '@frost-beta/mlx';
 *
 * const model = nn.Linear(10, 10);
 * console.log(Object.keys(model.parameters()));
 * // Outputs: ['weight', 'bias']
 *
 * // Square the parameters.
 * model.update(utils.treeMap((x: mx.array) => mx.multiply(x, x), model.parameters()));
 * ```
 *
 * @param fn - The function that processes the leaves of the tree.
 * @param tree - The main Object that will be iterated upon.
 * @param rest - Extra trees to be iterated together with `tree`.
 * @param isLeaf - An optional function that returns `true` if the passed
 * object is considered a leaf or `False` otherwise.
 *
 * @returns An Object with the new values returned by `fn`.
 */
export function treeMap(fn: (...args: unknown[]) => unknown,
                        tree: unknown,
                        rest?: object[],
                        isLeaf?: (node: unknown) => boolean): unknown {
  if (isLeaf && isLeaf(tree)) {
    return fn(tree, ...rest);
  } else if (Array.isArray(tree)) {
    return tree.map((child, i) => treeMap(fn, child, rest?.map(r => r[i]), isLeaf));
  } else if (typeof tree === 'object' && isDict(tree)) {
    const newTree = {};
    for (const k in tree) {
      newTree[k] = treeMap(fn, tree[k], rest?.map(r => r[k]), isLeaf);
    }
    return newTree;
  } else {
    if (rest)
      return fn(tree, ...rest);
    else
      return fn(tree);
  }
}

/**
 * Applies `fn` to the path and leaves of the Object `tree` and
 * returns a new Array with the results.
 *
 * @remarks
 *
 * This function is the same as `treeMap` but the `fn` takes the path as
 * the first argument followed by the remaining tree nodes.
 *
 * @param fn - The function that processes the leaves of the tree.
 * @param tree - The main Object that will be iterated upon.
 * @param rest - Extra trees to be iterated together with `tree`.
 * @param isLeaf - An optional function that returns `True` if the passed
 * object is considered a leaf or `False` otherwise.
 * @param path - The path to the current node.
 *
 * @returns An Object with the new values returned by `fn`.
 *
 * @example
 * ```ts
 * import {utils} from '@frost-beta/mlx';
 *
 * const tree = {model: [{w: 0, b: 1}, {w: 0, b: 1}]};
 * utils.treeMapWithPath((path: string) => console.log(path), tree);
 * // Outputs: model.0.w, model.0.b, model.1.w, model.1.b
 * ```
 */
export function treeMapWithPath(fn: (path?: string, ...args: unknown[]) => unknown,
                                tree: unknown,
                                rest?: object[],
                                isLeaf?: (node: unknown) => boolean,
                                path?: string): unknown {
  if (isLeaf && isLeaf(tree)) {
    if (rest)
      return fn(path, tree, ...rest);
    else
      return fn(path, tree);
  } else if (Array.isArray(tree)) {
    return tree.map((child, i) => treeMapWithPath(fn, child, rest?.map(r => r[i]), isLeaf, path ? `${path}.${i}` : `${i}`));
  } else if (typeof tree === 'object' && isDict(tree)) {
    const newTree = {};
    for (const k in tree) {
      newTree[k] = treeMapWithPath(fn, tree[k], rest?.map(r => r[k]), isLeaf, path ? `${path}.${k}` : `${k}`);
    }
    return newTree;
  } else {
    if (rest)
      return fn(path, tree, ...rest);
    else
      return fn(path, tree);
  }
}

/**
 * Flattens an Object tree to a list of key, value tuples.
 *
 * @remarks
 *
 * The keys are using the dot notation to define trees of arbitrary depth and
 * complexity.
 *
 *
 * @param tree - The Object tree to be flattened.
 * @param prefix - A prefix to use for the keys. The first character is always
 * discarded.
 * @param isLeaf - An optional function that returns true if the passed object
 * is considered a leaf or false otherwise.
 *
 * @returns An array of objects with `key` and `value` properties.
 *
 * @example
 * ```ts
 * import {utils} from '@frost-beta/mlx';
 *
 * console.log(utils.treeFlatten([[[0]]]));
 * // Outputs: [['0.0.0', 0]]
 *
 * console.log(utils.treeFlatten([[[0]]], '.hello'));
 * // Outputs: [['hello.0.0.0', 0]]
 * ```
 */
export function treeFlatten(tree: unknown,
                            prefix: string = '',
                            isLeaf?: (node: unknown) => boolean): [string, unknown][] {
  if (!isLeaf || !isLeaf(tree)) {
    let flatTree: [string, unknown][] = [];
    if (Array.isArray(tree)) {
      for (let i = 0; i < tree.length; i++) {
        flatTree = flatTree.concat(treeFlatten(tree[i], `${prefix}.${i}`, isLeaf));
      }
      return flatTree;
    } else if (typeof tree === 'object' && isDict(tree)) {
      for (let k in tree) {
        flatTree = flatTree.concat(treeFlatten(tree[k], `${prefix}.${k}`, isLeaf));
      }
      return flatTree;
    }
  }
  return [[prefix?.slice(1), tree]];
}

/**
 * Recreate an Object from its flat representation.
 *
 * @param tree - The flat representation of an Object. For instance as returned
 * by `treeFlatten`.
 *
 * @returns An Object.
 *
 * @example
 * ```ts
 * import { utils } from '@frost-beta/mlx';
 *
 * const d = utils.treeUnflatten([['hello.world', 42]]);
 * console.log(d);
 * // Outputs: {hello: {world: 42}}
 * ```
 */
export function treeUnflatten(tree: [string, unknown][]): unknown {
  if (tree.length === 1 && tree[0][0] === '') {
    return tree[0][1];
  }

  const isList = !Number.isNaN(parseInt(tree[0][0].split('.', 1)[0]));

  // Walkthrough path and collect children.
  const children: {[key: string]: [string, unknown][]} = {};
  for (let [key, value] of tree) {
    const [currentIndex, ...nextIndices] = key.split('.');
    const nextIndex = nextIndices.length === 0 ? '' : nextIndices[0];
    if (!(currentIndex in children)) {
      children[currentIndex] = [];
    }
    children[currentIndex].push([nextIndex, value]);
  }

  // Recursively map them to the original container.
  if (isList) {
    const keys = Object.keys(children).sort().map((idx) => parseInt(idx));
    return keys.map((i: number) => treeUnflatten(children[i]));
  } else {
    const newTree = {};
    for (let k in children) {
      newTree[k] = treeUnflatten(children[k]);
    }
    return newTree;
  }
}

// Internal helper to check if an object is a dictionary.
// Note that we are requiring the caller to do the comparison:
// typeof dict === 'object'
// In this way TypeScript can recognize the value as object instead of unknown
// in the caller's code.
export function isDict(dict: object) {
  if (dict === null)
    return false;
  // A plain object literal should have Object as its constructor.
  if (dict.constructor === Object)
    return true;
  // In Python the Module inherits from dict, there is no such concept in JS so
  // we directly check if dict is a Module instance.
  const {Module} = require('./nn/layers/base');
  return dict instanceof Module;
}