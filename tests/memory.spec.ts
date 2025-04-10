import {core as mx} from '..';
import {assert} from 'chai';

describe('memory', () => {
  it('memoryInfo', function() {
    if (!mx.metal.isAvailable())
      this.skip();

    this.timeout(10_000);
    let a = mx.zeros([4096], mx.int32);
    mx.eval(a);
    mx.synchronize();
    const activeMem = mx.getActiveMemory();
    assert.isAtLeast(activeMem, 4096 * 4);

    const b = mx.zeros([4096], mx.int32);
    mx.eval(b);
    mx.synchronize();

    const newActiveMem = mx.getActiveMemory();
    assert.isAbove(newActiveMem, activeMem);
    const peakMem = mx.getPeakMemory();
    assert.isAtLeast(peakMem, 4096 * 8);

    if (mx.metal.isAvailable()) {
      const cacheMem = mx.getCacheMemory();
      assert.isAtLeast(cacheMem, 4096 * 4);
    }

    mx.clearCache();
    assert.equal(mx.getCacheMemory(), 0);

    mx.resetPeakMemory();
    assert.equal(mx.getPeakMemory(), 0);
  });

  it('wiredMemory', function() {
    if (!mx.metal.isAvailable())
      this.skip();

    let oldLimit = mx.setWiredLimit(1000);
    oldLimit = mx.setWiredLimit(0);
    assert.equal(oldLimit, 1000);

    const maxSize = mx.metal.deviceInfo().max_recommended_working_set_size;
    assert.throws(() => mx.setWiredLimit(maxSize + 10));
  });
});
