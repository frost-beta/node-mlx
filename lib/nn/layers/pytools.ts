export function deepEqual(s1: number[], s2: number[]): boolean {
  return s1.length === s2.length && s1.every((u, i) => u === s2[i]);
}

export function range(start: number, end: number, step = 1): number[] {
  return Array.from({length: Math.ceil((end - start) / step)},
                    (_, i) => start + i * step);
}

// itertools.accumulate
// https://syakoo-lab.com/writings/20210723
export function* accumulate<T>(iterable: Iterable<T>, func: (x: T, y: T) => T, initial?: T) {
  let x = initial;
  if (x !== undefined)
    yield x;

  for (let y of iterable) {
    x = x === undefined ? y : func(x, y);
    yield x;
  }
}

// itertools.product
// https://gist.github.com/cybercase/db7dde901d7070c98c48
export function* product<T extends Array<Iterable<any>>>(...iterables: T): IterableIterator<{
  [K in keyof T]: T[K] extends Iterable<infer U> ? U : never
}> {
  if (iterables.length === 0)
    return;
  // Make a list of iterators from the iterables.
  const iterators = iterables.map(it => it[Symbol.iterator]());
  const results = iterators.map(it => it.next());
  if (results.some(r => r.done))
    throw new Error("Input contains an empty iterator.");
  for (let i = 0;;) {
    if (results[i].done) {
      // Reset the current iterator.
      iterators[i] = iterables[i][Symbol.iterator]();
      results[i] = iterators[i].next();
      // Advance, and exit if we've reached the end.
      if (++i >= iterators.length)
        return;
    } else {
      yield results.map(({value}) => value) as any;
      i = 0;
    }
    results[i] = iterators[i].next();
  }
}
