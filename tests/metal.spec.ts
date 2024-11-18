import {core as mx} from '..';
import {assert} from 'chai';

describe('metal', () => {
  before(function() {
    if (!mx.metal.isAvailable())
      this.skip();
  });

  it('memoryInfo', function() {
    this.timeout(10_000);
    let a = mx.zeros([4096], mx.int32);
    mx.eval(a);
    mx.synchronize();
    const activeMem = mx.metal.getActiveMemory();
    assert.isAtLeast(activeMem, 4096 * 4);

    const b = mx.zeros([4096], mx.int32);
    mx.eval(b);
    mx.synchronize();

    const newActiveMem = mx.metal.getActiveMemory();
    assert.isAbove(newActiveMem, activeMem);
    const peakMem = mx.metal.getPeakMemory();
    assert.isAtLeast(peakMem, 4096 * 8);

    mx.metal.clearCache();
    assert.equal(mx.metal.getCacheMemory(), 0);

    mx.metal.resetPeakMemory();
    assert.equal(mx.metal.getPeakMemory(), 0);

    let oldLimit = mx.metal.setWiredLimit(1000);
    oldLimit = mx.metal.setWiredLimit(0);
    assert.equal(oldLimit, 1000);

    const maxSize = mx.metal.deviceInfo().max_recommended_working_set_size;
    assert.throws(() => mx.metal.setWiredLimit(maxSize + 10));
  });
});
