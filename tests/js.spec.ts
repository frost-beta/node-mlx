import {core as mx} from '..';
import {assert} from 'chai';

describe('js', () => {
  describe('constructor', () => {
    it('sparseArray', () => {
      const sparse = mx.array(new Array(10));
      const zeros = mx.zeros([10]);
      assert.deepEqual(sparse.tolist(), zeros.tolist());
    });

    it('buffer', () => {
      const buffer = Buffer.from([8, 9, 6, 4]);
      const array = mx.array(buffer);
      assert.equal(array.dtype, mx.uint8);
      assert.equal(array.size, buffer.length);
      assert.deepEqual(array.tolist(), [...buffer]);
    });

    it('typedArray', () => {
      const buffer = new Int8Array(new ArrayBuffer(10));
      const array = mx.array(buffer);
      assert.equal(array.dtype, mx.int8);
      assert.equal(array.size, buffer.byteLength);
      assert.deepEqual(array.tolist(), [...Array.from(buffer)]);
    });
  });

  describe('toString', () => {
    it('array', () => {
      assert.equal(mx.array([1, 2, 3, 4]).toString(),
                   'array([1, 2, 3, 4], dtype=float32)');
    });
  });

  describe('tidy', () => {
    it('unwrapObjects', () => {
      const objects: mx.array[] = [];
      mx.tidy(() => {
        for (let i = 0; i < 100; ++i) {
          objects.push(mx.array([8964]));
        }
      });
      for (let i = 0; i < 100; ++i) {
        assert.throws(() => objects[i].size, 'Error converting "this" to mx.array.');
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
      const objects: mx.array[] = [];
      mx.tidy(() => {
        for (let i = 0; i < 99; ++i) {
          objects.push(mx.array([8964]));
        }
        let ret = mx.array([8964]);
        objects.push(ret);
        return ret;
      });
      for (let i = 0; i < 99; ++i) {
        assert.throws(() => objects[i].size, 'Error converting "this" to mx.array.');
      }
      assert.equal(objects[99].item(), 8964);
    });

    it('await', async () => {
      const objects: mx.array[] = [];
      await mx.tidy(async () => {
        for (let i = 0; i < 100; ++i) {
          objects.push(mx.array([8964]));
          await new Promise(resolve => process.nextTick(resolve));
        }
      });
      for (let i = 0; i < objects.length; ++i) {
        assert.throws(() => objects[i].size, 'Error converting "this" to mx.array.');
      }
    });

    it('awaitReturnValue', async () => {
      const b = await mx.tidy(() => {
        const a = mx.array([8, 9, 6, 4]);
        return new Promise<mx.array>(resolve => process.nextTick(() => resolve(a)));
      });
      assert.deepEqual(b.tolist(), [8, 9, 6, 4]);
    });

    it('awaitNested', async () => {
      const intermediate: mx.array[] = [];
      const b = await mx.tidy(async () => {
        intermediate.push(mx.array([1]));
        await new Promise(resolve => process.nextTick(resolve));
        return await mx.tidy(() => {
          intermediate.push(mx.array([2]));
          const a = mx.array([8, 9, 6, 4]);
          return new Promise<mx.array>(resolve => process.nextTick(() => resolve(a)));
        });
      });
      assert.equal(intermediate.length, 2);
      for (const i of intermediate) {
        assert.throws(() => i.size, 'Error converting "this" to mx.array.');
      }
      assert.deepEqual(b.tolist(), [8, 9, 6, 4]);
    });
  });

  describe('dispose', () => {
    it('nested', () => {
      const obj = {a: mx.array([1, 2, 3, 4])};
      mx.dispose(obj);
      assert.throws(() => obj.a.size, 'Error converting "this" to mx.array.');
    });

    it('args', () => {
      const a = mx.array([0]);
      const b = mx.array([0]);
      mx.dispose(a, b);
      assert.throws(() => a.size, 'Error converting "this" to mx.array.');
      assert.throws(() => b.size, 'Error converting "this" to mx.array.');
    });
  });
});
