import mx from '..';
import {assert} from 'chai';

export function assertArray(a: mx.array, assertion: (arrays: boolean[]) => void) {
  assert.isTrue(a instanceof mx.array);
  assert.equal(a.dtype, mx.bool_);
  if (a.ndim == 0) {
    assertion([ a.item() as boolean ]);
  } else {
    const list = a.tolist();
    assertion(list as boolean[]);
  }
}

export const assertArrayAllTrue = (a) => assertArray(a, (arrays) => assert.notInclude(arrays, false));
export const assertArrayAllFalse = (a) => assertArray(a, (arrays) => assert.notInclude(arrays, true));

export function* permutations(array: any[], num: number) {
  const copy = array.slice();
  for (let i = 0; i < num && copy.length; i++) {
    const remaining = copy.slice();
    const element = remaining.splice(i, 1);
    for (let subArray of permutations(remaining, num - 1)) {
      yield [element, ...subArray];
    }
  }
}
