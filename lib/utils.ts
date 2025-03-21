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
 * ```typescript
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
 * @param func - The function that processes the leaves of the tree.
 * @param tree - The main Object that will be iterated upon.
 * @param rest - Extra trees to be iterated together with `tree`.
 * @param isLeaf - An optional function that returns `true` if the passed
 * object is considered a leaf or `False` otherwise.
 *
 * @returns An Object with the new values returned by `fn`.
 */
export function treeMap(func: (...args: any[]) => unknown,
                        tree: Record<string, any>,
                        rest?: Record<string, any>[],
                        isLeaf?: (node: unknown) => boolean): Record<string, any> {
  if (isLeaf && isLeaf(tree)) {
    if (rest)
      return func(tree, ...rest);
    else
      return func(tree);
  } else if (Array.isArray(tree)) {
    return tree.map((child, i) => treeMap(func, child, rest?.map(r => r[i]), isLeaf));
  } else if (typeof tree === 'object' && isDict(tree)) {
    const newTree: Record<string, any> = {};
    for (const k in tree) {
      newTree[k] = treeMap(func, tree[k], rest?.map(r => r[k]), isLeaf);
    }
    return newTree;
  } else {
    if (rest)
      return func(tree, ...rest);
    else
      return func(tree);
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
 * ```typescript
 * import {utils} from '@frost-beta/mlx';
 *
 * const tree = {model: [{w: 0, b: 1}, {w: 0, b: 1}]};
 * utils.treeMapWithPath((path: string) => console.log(path), tree);
 * // Outputs: model.0.w, model.0.b, model.1.w, model.1.b
 * ```
 */
export function treeMapWithPath(fn: (path?: string, ...args: unknown[]) => unknown,
                                tree: Record<string, any>,
                                rest?: Record<string, any>[],
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
    const newTree: Record<string, any> = {};
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
 * @param convertKey - An optional function that converts the key of tree.
 *
 * @returns An array of objects with `key` and `value` properties.
 *
 * @example
 * ```typescript
 * import {utils} from '@frost-beta/mlx';
 *
 * console.log(utils.treeFlatten([[[0]]]));
 * // Outputs: [['0.0.0', 0]]
 *
 * console.log(utils.treeFlatten([[[0]]], '.hello'));
 * // Outputs: [['hello.0.0.0', 0]]
 * ```
 */
export function treeFlatten(tree: Record<string, any>,
                            prefix: string = '',
                            isLeaf?: (node: unknown) => boolean,
                            convertKey?: (key: string) => string): [string, unknown][] {
  if (!isLeaf || !isLeaf(tree)) {
    let flatTree: [string, unknown][] = [];
    if (Array.isArray(tree)) {
      for (let i = 0; i < tree.length; i++) {
        flatTree = flatTree.concat(treeFlatten(tree[i], `${prefix}.${i}`, isLeaf, convertKey));
      }
      return flatTree;
    } else if (typeof tree === 'object' && isDict(tree)) {
      for (let k in tree) {
        const key = convertKey ? convertKey(k) : k;
        flatTree = flatTree.concat(treeFlatten(tree[k], `${prefix}.${key}`, isLeaf, convertKey));
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
 * @param convertKey - An optional function that converts the key of tree.
 *
 * @returns An Object.
 *
 * @example
 * ```typescript
 * import {utils} from '@frost-beta/mlx';
 *
 * const d = utils.treeUnflatten([['hello.world', 42]]);
 * console.log(d);
 * // Outputs: {hello: {world: 42}}
 * ```
 */
export function treeUnflatten(tree: [string, unknown][],
                              convertKey?: (key: string) => string): unknown {
  if (tree.length === 1 && tree[0][0] === '') {
    return tree[0][1];
  }

  const isList = !Number.isNaN(parseInt(tree[0][0].split('.', 1)[0]));

  // Walkthrough path and collect children.
  const children: {[key: string]: [string, unknown][]} = {};
  for (let [key, value] of tree) {
    const [index, ...nextIndices] = key.split('.');
    const next = nextIndices?.join('.') ?? '';
    if (!(index in children))
      children[index] = [];
    children[index].push([next, value]);
  }

  // Recursively map them to the original container.
  if (isList) {
    const keys = Object.keys(children).sort().map((idx) => [ parseInt(idx), idx ]);
    const newList: any[] = [];
    for (const [i, k] of keys)
      newList[i as number] = treeUnflatten(children[k], convertKey);
    return newList;
  } else {
    const newTree: Record<string, any> = {};
    for (let k in children) {
      const key = convertKey ? convertKey(k) : k;
      newTree[key] = treeUnflatten(children[k], convertKey);
    }
    return newTree;
  }
}

/**
 * Applies a reduction to the leaves of a tree.
 *
 * @remarks
 *
 * This function reduces trees into an accumulated result by applying
 * the provided function `func` to the leaves of the tree.
 *
 * @example
 * ```
 * const tree = {a: [1, 2, 3], b: [4, 5]};
 * treeReduce((acc, x) => acc + x, tree, 0);  // Returns 15
 * ```
 *
 * @param func - The reducer function that takes two arguments (accumulator,
 * current value) and returns the updated accumulator.
 * @param tree - The Python tree to reduce. It can be any nested combination of
 * lists, tuples, or dictionaries.
 * @param initializer - The initial value to start the reduction. If
 * not provided, the first leaf value is used.
 * @param isLeaf - A function to determine if an object is a
 * leaf, returning `true` for leaf nodes and `false` otherwise.
 *
 * @returns The accumulated value.
 */
export function treeReduce<U>(func: (accumulator: U, currentValue: U) => U,
                              tree: Nested<U>,
                              initializer?: U,
                              isLeaf?: (node: unknown) => boolean): U {
  if (isLeaf && isLeaf(tree))
    return initializer != null ? func(initializer, tree as U) : tree as U;

  let accumulator = initializer;

  if (Array.isArray(tree)) {
    for (const item of tree)
      accumulator = treeReduce(func, item, accumulator, isLeaf);
  } else if (typeof tree === 'object' && isDict(tree)) {
    for (const item of Object.values(tree))
      accumulator = treeReduce(func, item, accumulator, isLeaf);
  } else {
    return accumulator != null ? func(accumulator, tree as U) : tree as U;
  }

  return accumulator;
}

/**
 * Merges two trees into one containing the values of both.
 * It can be considered as a deep dict.update method.
 *
 * @param treeA - The first tree.
 * @param treeB - The second tree.
 * @param mergeFn - An optional function to merge leaves.
 *
 * @returns The tree containing the values of both `treeA` and `treeB`.
 */
export function treeMerge<A, B>(treeA: A, treeB: B, mergeFn?: (a: A, b: B) => A | B): A | B {
  if ((Array.isArray(treeA) && treeA.length === 0) ||
      (typeof treeA === 'object' && treeA !== null && Object.keys(treeA).length === 0)) {
    treeA = null;
  }
  if ((Array.isArray(treeB) && treeB.length == 0) ||
      (typeof treeB === 'object' && treeB !== null && Object.keys(treeB).length === 0)) {
    treeB = null;
  }
  if (treeA == null && treeB != null) return treeB;
  if (treeA != null && treeB == null) return treeA;

  if (Array.isArray(treeA) && Array.isArray(treeB)) {
    let zipped = zipLongest(treeA, treeB);
    return zipped.map(([a, b]) => treeMerge(a, b, mergeFn)) as A | B;
  } else if (typeof treeA === 'object' && typeof treeB === 'object') {
    let keys = new Set([...Object.keys(treeA), ...Object.keys(treeB)]);
    let merged: any = {};
    for (let k of keys)
      merged[k] = treeMerge((treeA as any)[k], (treeB as any)[k], mergeFn);
    return merged;
  } else {
    if (!mergeFn) {
      throw new Error('Trees contain elements at the same locations but no merge function was provided');
    }
    return mergeFn(treeA, treeB);
  }
}

// Internal helper to zip two arrays of different lengths.
function zipLongest<T, U>(arr1: T[], arr2: U[]): [T, U][] {
  const length = Math.max(arr1.length, arr2.length);
  let result: [T, U][] = [];
  for (let i = 0; i < length; i++)
    result.push([arr1[i], arr2[i]]);
  return result;
}

// A nested type that always has T as leaves.
export type Nested<T> = T | T[] | {[key: string]: Nested<T>};
export type NestedDict<T> = {[key: string]: Nested<T>};

// Compare if 2 arrays are equal.
// TOOD(zcbenz): Extend to objects when needed.
export function deepEqual(s1: number[], s2: number[]): boolean {
  return s1.length === s2.length && s1.every((u, i) => u === s2[i]);
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

export function toCamelCase(e: string) {
  return e.replace(/_([a-z])/g, (g) =>  g[1].toUpperCase());
}

export function toSnakeCase(e: string) {
  // Do not convert if the first character is upper case.
  if (e.length > 0 && e[0] == e[0].toUpperCase())
    return e;
  return e.replace(/[A-Z]/g, (ch, i) => i ? '_' + ch.toLowerCase() : ch.toLowerCase());
}
