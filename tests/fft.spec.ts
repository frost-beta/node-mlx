import {core as mx} from '..';
import {assert} from 'chai';

// TODO(zcbenz): The tests were written by ChatGPT, rewrite them.
describe('fft', () => {
  let useCpu: Disposable;
  before(() => useCpu = mx.stream(mx.cpu));
  after(() => useCpu[Symbol.dispose]());

  it('fft', () => {
    const input = mx.array([0, 1, 2, 3]);
    const result = mx.fft.fft(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [6, -2, -2, -2]);
  });

  it('ifft', () => {
    const input = mx.array([0, 1, 2, 3]);
    const result = mx.fft.ifft(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [1.5, -0.5, -0.5, -0.5]);
  });

  it('fft2', () => {
    const input = mx.array([[1, 2], [3, 4]]);
    const result = mx.fft.fft2(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[10, -2], [-4, 0]]);
  });

  it('ifft2', () => {
    const input = mx.array([[1, 2], [3, 4]]);
    const result = mx.fft.ifft2(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[10, -2], [-4, 0]]);
  });

  it('fftn', () => {
    const input = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const result = mx.fft.fftn(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[[36, -4], [-8, 0]], [[-16, 0], [0, 0]]]);
  });

  it('ifftn', () => {
    const input = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const result = mx.fft.ifftn(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[[36, -4], [-8, 0]], [[-16, 0], [0, 0]]]);
  });

  it('rfft', () => {
    const input = mx.array([0, 1, 2, 3]);
    const result = mx.fft.rfft(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [6, -2, -2]);
  });

  it('irfft', () => {
    const input = mx.array([0, 1, 2]);
    const result = mx.fft.irfft(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [1, -0.5, 0, -0.5]);
  });

  it('rfft2', () => {
    const input = mx.array([[1, 2], [3, 4]]);
    const result = mx.fft.rfft2(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[10, -2], [-4, 0]]);
  });

  it('irfft2', () => {
    const input = mx.array([[1, 2], [3, 4]]);
    const result = mx.fft.irfft2(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[10, -2], [-4, 0]]);
  });

  it('rfftn', () => {
    const input = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const result = mx.fft.rfftn(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[[36, -4], [-8, 0]], [[-16, 0], [0, 0]]]);
  });

  it('irfftn', () => {
    const input = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const result = mx.fft.irfftn(input);
    assert.deepEqual(result.astype(mx.float32).tolist(), [[[36, -4], [-8, 0]], [[-16, 0], [0, 0]]]);
  });
});
