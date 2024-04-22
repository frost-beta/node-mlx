import mx from '..';
import {assert} from 'chai';

describe('Metal', () => {
  before(function() {
    if (!mx.metal.isAvailable())
      this.skip();
  });

  it('memoryInfo', () => {
    let a = mx.zeros([4096], mx.int32, mx.cpu);
    mx.eval(a);
    const activeMem = mx.metal.getActiveMemory();
    assert.isAtLeast(activeMem, 4096 * 4);

    const b = mx.zeros([4096], mx.int32, mx.cpu);
    mx.eval(b);

    const newActiveMem = mx.metal.getActiveMemory();
    assert.isAbove(newActiveMem, activeMem);
    const peakMem = mx.metal.getPeakMemory();
    assert.isAtLeast(peakMem, 4096 * 8);
  });
});
