import {core as mx} from '..';
import {assert} from 'chai';

describe('js', () => {
  describe('toString', () => {
    it('array', () => {
      assert.equal(mx.array([1, 2, 3, 4]), 'array([1, 2, 3, 4], dtype=float32)');
    });
  });

  describe('tidy', () => {
    it('unwrapObjects', () => {
      let objects = [];
      mx.tidy(() => {
        for (let i = 0; i < 100; ++i) {
          objects.push(mx.array([8964]));
        }
      });
      for (let i = 0; i < 100; ++i) {
        assert.throws(() => objects[i].size, 'Error converting "this" to array.');
      }
    });

    it('keepReturnedArray', () => {
      let a;
      let b;
      b = mx.tidy(() => {
        a = mx.array([8, 9, 6, 4]);
        return a;
      });
      assert.equal(a, b);
      assert.deepEqual(a.tolist(), [8, 9, 6, 4]);
    });

    it('keepNestedArray', () => {
      let objects = [];
      mx.tidy(() => {
        for (let i = 0; i < 99; ++i) {
          objects.push(mx.array([8964]));
        }
        let ret = mx.array([8964]);
        objects.push(ret);
        return ret;
      });
      for (let i = 0; i < 99; ++i) {
        assert.throws(() => objects[i].size, 'Error converting "this" to array.');
      }
      assert.equal(objects[99].item(), 8964);
    });
  });

  describe('dispose', () => {
    it('nested', () => {
      const obj = {a: mx.array([1, 2, 3, 4])};
      mx.dispose(obj);
      assert.throws(() => obj.a.size, 'Error converting "this" to array.');
    });

    it('args', () => {
      const a = mx.array([0]);
      const b = mx.array([0]);
      mx.dispose(a, b);
      assert.throws(() => a.size, 'Error converting "this" to array.');
      assert.throws(() => b.size, 'Error converting "this" to array.');
    });
  });
});
