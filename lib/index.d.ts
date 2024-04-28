// Device.
export type DeviceType = number;

export const cpu: DeviceType;
export const gpu: DeviceType;

export class Device {
  type: DeviceType;
}

type DeviceOrType = Device | DeviceType;

export function defaultDevice(): Device;
export function setDefaultDevice(device: DeviceOrType);

// Stream.
export class Stream {
  device: Device;
}

export function defaultStream(device: DeviceOrType): Stream;
export function setDefaultStream();
export function newStream(device: DeviceOrType): Stream;
export function toStream(s: StreamOrDevice): Stream;
export function stream(s: StreamOrDevice): Disposable;

export type StreamOrDevice = Stream | Device | DeviceType | undefined;

// Dtype.
export class Dtype {
  size: number;
}

export const bool: Dtype;
export const bool_: Dtype;
export const uint8: Dtype;
export const uint16: Dtype;
export const uint32: Dtype;
export const uint64: Dtype;
export const int8: Dtype;
export const int16: Dtype;
export const int32: Dtype;
export const int64: Dtype;
export const float16: Dtype;
export const float32: Dtype;
export const bfloat16: Dtype;
export const complex64: Dtype;

export type DtypeCategory = number;

export const complexfloating: DtypeCategory
export const floating: DtypeCategory
export const inexact: DtypeCategory
export const signedinteger: DtypeCategory
export const unsignedinteger: DtypeCategory
export const integer: DtypeCategory
export const number: DtypeCategory
export const generic: DtypeCategory

// Complex number.
interface Complex {
  re: number;
  im?: number;
}
export function Complex(real: number, imag?: number): Complex;

// Index slice.
interface Slice {
  start: number | null;
  stop: number | null;
  step: number | null;
}
export function Slice(start?: number, stop?: number, step?: number): Slice;

// Array helper types.
type MultiDimensionalArray<T> = MultiDimensionalArray<T>[] | T;
type Scalar = boolean | number | Complex;
type ScalarOrArray = MultiDimensionalArray<Scalar | array>;
type ArrayIndex = null | Slice | '...' | array | number;

// Helper class to apply updates at specific indices.
declare class ArrayAt {
  constructor(array: array, indices: ArrayIndex | ArrayIndex[]);

  add(value: ScalarOrArray): array;
  subtract(value: ScalarOrArray): array;
  multiply(value: ScalarOrArray): array;
  divide(value: ScalarOrArray): array;
  maximum(value: ScalarOrArray): array;
  minimum(value: ScalarOrArray): array;
}

// Array.
export function array(value: ScalarOrArray, dtype?: Dtype): array;

export class array {
  constructor(value: ScalarOrArray, dtype?: Dtype);

  length: number;
  astype(dtype: Dtype, s?: StreamOrDevice): array;
  at(...index: ArrayIndex[]): ArrayAt;
  item(): Scalar;
  tolist(): MultiDimensionalArray<Scalar>;
  dtype: Dtype;
  itemsize: number;
  nbytes?: number;
  ndim: number;
  shape: number[];
  size: number;
  T: array;

  abs(s?: StreamOrDevice): array;
  all(keepdims?: boolean, s?: StreamOrDevice): array;
  all(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  any(keepdims?: boolean, s?: StreamOrDevice): array;
  any(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  argmax(keepdims?: boolean, s?: StreamOrDevice): array;
  argmax(axis?: number, keepdims?: boolean, s?: StreamOrDevice): array;
  argmin(keepdims?: boolean, s?: StreamOrDevice): array;
  argmin(axis?: number, keepdims?: boolean, s?: StreamOrDevice): array;
  cos(s?: StreamOrDevice): array;
  cummax(s?: StreamOrDevice): array;
  cummin(s?: StreamOrDevice): array;
  cumprod(s?: StreamOrDevice): array;
  cumsum(s?: StreamOrDevice): array;
  diag(k?: number, s?: StreamOrDevice): array;
  diagonal(offset?: number, axis1?: number, axis2?: number, s?: StreamOrDevice): array;
  exp(s?: StreamOrDevice): array;
  power(exponent: ScalarOrArray, s?: StreamOrDevice): array;
  flatten(startAxis?: number, endAxis?: number, s?: StreamOrDevice): array;
  log(s?: StreamOrDevice): array;
  log10(s?: StreamOrDevice): array;
  log1p(s?: StreamOrDevice): array;
  log2(s?: StreamOrDevice): array;
  logsumexp(s?: StreamOrDevice): array;
  max(keepdims?: boolean, s?: StreamOrDevice): array;
  max(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  mean(keepdims?: boolean, s?: StreamOrDevice): array;
  mean(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  min(keepdims?: boolean, s?: StreamOrDevice): array;
  min(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  moveaxis(source: number, destination: number, s?: StreamOrDevice): array;
  prod(s?: StreamOrDevice): array;
  reciprocal(s?: StreamOrDevice): array;
  reshape(shape: number | number[], s?: StreamOrDevice): array;
  round(s?: StreamOrDevice): array;
  rsqrt(s?: StreamOrDevice): array;
  sin(s?: StreamOrDevice): array;
  split(indicesOrSections?: number | number[], s?: StreamOrDevice): array[];
  sqrt(s?: StreamOrDevice): array;
  square(s?: StreamOrDevice): array;
  squeeze(axis?: number | number[], s?: StreamOrDevice): array;
  swapaxes(axis1: number, axis2: number, s?: StreamOrDevice): array;
  sum(s?: StreamOrDevice): array;
  transpose(s?: StreamOrDevice): array;
  variance(indicesOrSections?: number | number[], keepdims?: boolean, ddof?: number, s?: StreamOrDevice): array;

  index(...index: ArrayIndex[]): array;
  indexPut_(index: ArrayIndex | ArrayIndex[], value: ScalarOrArray): array;
  [Symbol.iterator](): IterableIterator<array>;
}

// Ops.
export function abs(array: ScalarOrArray, s?: StreamOrDevice): array;
export function add(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function all(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function all(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function allclose(array1: ScalarOrArray, array2: ScalarOrArray, rtol?: number, atol?: number, equalNan?: boolean, s?: StreamOrDevice): boolean;
export function any(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function any(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function arange(start: number, stop?: number, step?: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function arange(stop: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function arccos(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arccosh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arcsin(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arcsinh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arctan(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arctanh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function argmax(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function argmax(array: ScalarOrArray, axis?: number, keepdims?: boolean, s?: StreamOrDevice): array;
export function argmin(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function argmin(array: ScalarOrArray, axis?: number, keepdims?: boolean, s?: StreamOrDevice): array;
export function argpartition(array: ScalarOrArray, kth: number, axis?: number, s?: StreamOrDevice): array;
export function argsort(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arrayEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function asStrided(array: ScalarOrArray, shape?: number[], strides?: number[], offset?: number, s?: StreamOrDevice): array;
export function atleast1d(...arrays: array[]): array;
export function atleast2d(...arrays: array[]): array;
export function atleast3d(...arrays: array[]): array;
export function broadcastTo(array: ScalarOrArray, shape: number | number[], s?: StreamOrDevice): array;
export function ceil(array: ScalarOrArray, s?: StreamOrDevice): array;
export function clip(array: ScalarOrArray, min: ScalarOrArray, max: ScalarOrArray, s?: StreamOrDevice): array;
export function concatenate(arrays?: array[], axis?: number, s?: StreamOrDevice): array;
export function convolve(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function conv1d(input: ScalarOrArray, weights?: ScalarOrArray, s?: StreamOrDevice): array;
export function conv2d(input: ScalarOrArray, weights?: ScalarOrArray, s?: StreamOrDevice): array;
export function convGeneral(input: ScalarOrArray, weights?: ScalarOrArray, s?: StreamOrDevice): array;
export function cos(array: ScalarOrArray, s?: StreamOrDevice): array;
export function cosh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function cummax(array: ScalarOrArray, s?: StreamOrDevice): array;
export function cummin(array: ScalarOrArray, s?: StreamOrDevice): array;
export function cumprod(array: ScalarOrArray, s?: StreamOrDevice): array;
export function cumsum(array: ScalarOrArray, s?: StreamOrDevice): array;
export function dequantize(array: ScalarOrArray, s?: StreamOrDevice): array;
export function diag(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
export function diagonal(array: ScalarOrArray, offset?: number, axis1?: number, axis2?: number, s?: StreamOrDevice): array;
export function divide(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function divmod(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): [array, array];
export function equal(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function notEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function erf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function erfinv(array: ScalarOrArray, s?: StreamOrDevice): array;
export function exp(array: ScalarOrArray, s?: StreamOrDevice): array;
export function expm1(array: ScalarOrArray, s?: StreamOrDevice): array;
export function expandDims(array: ScalarOrArray, dims: number | number[], s?: StreamOrDevice): array;
export function eye(N: number, M?: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function flatten(array: ScalarOrArray, startAxis?: number, endAxis?: number, s?: StreamOrDevice): array;
export function floor(array: ScalarOrArray, s?: StreamOrDevice): array;
export function floorDivide(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function full(shape: number | number[], fillValue: ScalarOrArray, dtype?: Dtype, s?: StreamOrDevice): array;
export function greater(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function greaterEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function identity(n: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function inner(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function isclose(array1: ScalarOrArray, array2: ScalarOrArray, rtol?: number, atol?: number, equalNan?: boolean, s?: StreamOrDevice): array;
export function isinf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function isnan(array: ScalarOrArray, s?: StreamOrDevice): array;
export function isneginf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function isposinf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function less(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function lessEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function linspace(start: number, stop: number, num?: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function load(filepath: string, s?: StreamOrDevice): array;
export function log(array: ScalarOrArray, s?: StreamOrDevice): array;
export function log2(array: ScalarOrArray, s?: StreamOrDevice): array;
export function log10(array: ScalarOrArray, s?: StreamOrDevice): array;
export function log1p(array: ScalarOrArray, s?: StreamOrDevice): array;
export function logaddexp(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function logicalNot(array: ScalarOrArray, s?: StreamOrDevice): array;
export function logicalAnd(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function logicalOr(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function logsumexp(array: ScalarOrArray, s?: StreamOrDevice): array;
export function matmul(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function max(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function max(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function maximum(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function mean(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function mean(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function meshgrid(...arrays: array[]): array[];
export function min(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function min(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function minimum(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function moveaxis(array: ScalarOrArray, source: number | number[], destination: number | number[], s?: StreamOrDevice): array;
export function multiply(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function negative(array: ScalarOrArray, s?: StreamOrDevice): array;
export function ones(shape: number | number[], dtype?: Dtype, s?: StreamOrDevice): array;
export function onesLike(array: ScalarOrArray, s?: StreamOrDevice): array;
export function outer(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function partition(array: ScalarOrArray, kth: number, axis?: number, s?: StreamOrDevice): array;
export function pad(array: ScalarOrArray, padWidth: number | [number] | [number, number] | [number, number][], constantValue?: ScalarOrArray, s?: StreamOrDevice): array;
export function power(array: ScalarOrArray, exponent: ScalarOrArray, s?: StreamOrDevice): array;
export function prod(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function prod(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function quantize(array: ScalarOrArray, s?: StreamOrDevice): array;
export function quantizedMatmul(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function reciprocal(array: ScalarOrArray, s?: StreamOrDevice): array;
export function remainder(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function repeat(array: ScalarOrArray, repeats?: number, axis?: number, s?: StreamOrDevice): array;
export function reshape(array: ScalarOrArray, shape: number | number[], s?: StreamOrDevice): array;
export function round(array: ScalarOrArray, s?: StreamOrDevice): array;
export function rsqrt(array: ScalarOrArray, s?: StreamOrDevice): array;
export function save(array: ScalarOrArray, filepath: string, s?: StreamOrDevice): void;
export function savez(dict: { [key: string]: array }, filepath: string, s?: StreamOrDevice): void;
export function savezCompressed(dict: { [key: string]: array }, filepath: string, s?: StreamOrDevice): void;
export function saveGguf(array: ScalarOrArray, filepath: string, s?: StreamOrDevice): void;
export function saveSafetensors(dict: { [key: string]: array }, filepath: string, s?: StreamOrDevice): void;
export function sigmoid(array: ScalarOrArray, s?: StreamOrDevice): array;
export function sign(array: ScalarOrArray, s?: StreamOrDevice): array;
export function sin(array: ScalarOrArray, s?: StreamOrDevice): array;
export function sinh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function softmax(array: ScalarOrArray, s?: StreamOrDevice): array;
export function sort(array: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
export function split(array: ScalarOrArray, indicesOrSections?: number | number[], axis?: number, s?: StreamOrDevice): array[];
export function sqrt(array: ScalarOrArray, s?: StreamOrDevice): array;
export function square(array: ScalarOrArray, s?: StreamOrDevice): array;
export function squeeze(array: ScalarOrArray, axis?: number | number[], s?: StreamOrDevice): array;
export function stack(arrays?: array[], axis?: number, s?: StreamOrDevice): array;
export function std(array: ScalarOrArray, s?: StreamOrDevice): array;
export function stopGradient(array: ScalarOrArray, s?: StreamOrDevice): array;
export function subtract(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function sum(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
export function sum(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
export function swapaxes(array: ScalarOrArray, axis1: number, axis2: number, s?: StreamOrDevice): array;
export function take(array: ScalarOrArray, indices?: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
export function takeAlongAxis(array: ScalarOrArray, indices?: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
export function tan(array: ScalarOrArray, s?: StreamOrDevice): array;
export function tanh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function tensordot(array1: ScalarOrArray, array2: ScalarOrArray, axes?: number | [number | number[], number | number[]], s?: StreamOrDevice): array;
export function tile(array: ScalarOrArray, reps?: number[], s?: StreamOrDevice): array;
export function topk(array: ScalarOrArray, k: number, axis?: number, s?: StreamOrDevice): [array, array];
export function transpose(array: ScalarOrArray, axes?: number[], s?: StreamOrDevice): array;
export function tri(N: number, M?: number, k?: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function tril(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
export function triu(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
export function variance(array: ScalarOrArray, indicesOrSections?: number | number[], keepdims?: boolean, ddof?: number, s?: StreamOrDevice): array;
export function where(condition: ScalarOrArray, x: ScalarOrArray, y: ScalarOrArray, s?: StreamOrDevice): array;
export function zeros(shape: number | number[], dtype?: Dtype, s?: StreamOrDevice): array;
export function zerosLike(array: ScalarOrArray, s?: StreamOrDevice): array;

// Transforms.
// @ts-ignore
export function eval(...args: ScalarOrArray[]): void;
export function asyncEval(...args: ScalarOrArray[]): void;
export function jvp(func: (...args: array[]) => array | array[], primals: array[], tangents: array[]): [array[], array[]];
export function vjp(func: (...args: array[]) => array | array[], primals: array[], cotangents: array[]): [array[], array[]];
type ValueAndGradFunctionScalar = (...args: array[]) => [array, array]
type ValueAndGradFunctionGeneric = (...args: array[]) => [array[], array[]]
export function valueAndGrad(func: (...args: array[]) => array, argnums?: number | number[]): ValueAndGradFunctionScalar;
export function valueAndGrad(func: (...args: array[]) => array[], argnums?: number | number[]): ValueAndGradFunctionGeneric;
type GradFunctionScalar = (...args: array[]) => array
type GradFunctionGeneric = (...args: array[]) => array[]
export function grad(func: (...args: array[]) => array, argnums?: number | number[]): GradFunctionScalar;
export function grad(func: (...args: array[]) => array[], argnums?: number | number[]): GradFunctionGeneric;

// Metal.
export namespace metal {
  export function isAvailable(): boolean;
  export function getActiveMemory(): number;
  export function getPeakMemory(): number;
  export function getCacheMemory(): number;
  export function setMemoryLimit(limit: number, relaxed?: boolean): number;
  export function setCacheLimit(limit: number): number;
  export function startCapture(path: string): boolean;
  export function stopCapture(): void;
}

// Random.
export namespace random {
  export function bernoulli(p?: ScalarOrArray, shape?: number[], key?: array, s?: StreamOrDevice): array;
  export function categorical(logits: ScalarOrArray, axis?: number, shape?: number[], numSamples?: number, key?: array, s?: StreamOrDevice): array;
  export function gumbel(shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
  export function key(seed: BigInt): any;
  export function normal(shape?: number[], dtype?: Dtype, loc?: number, scale?: number, key?: array, s?: StreamOrDevice): array;
  export function multivariateNormal(mean: array, cov: array, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
  export function randint(low: ScalarOrArray, high: ScalarOrArray, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
  export function seed(seed: BigInt): void;
  export function split(array: array, num?: number, s?: StreamOrDevice): array;
  export function truncatedNormal(lower: ScalarOrArray, upper: ScalarOrArray, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
  export function uniform(low: ScalarOrArray, high: ScalarOrArray, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
}

// FFT.
export namespace fft {
  export function fft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
  export function ifft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
  export function fft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function ifft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function fftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function ifftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function rfft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
  export function irfft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
  export function rfft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function irfft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function rfftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  export function irfftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
}

// Linear algebra.
export namespace linalg {
  export function norm(array: ScalarOrArray, norm?: number | string, axes?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  export function qr(array: ScalarOrArray, s?: StreamOrDevice): array[];
  export function svd(array: ScalarOrArray, s?: StreamOrDevice): array[];
  export function inv(array: ScalarOrArray, s?: StreamOrDevice): array;
}

// Fast operations.
export namespace fast {
  export function rmsNorm(array: ScalarOrArray, weights: ScalarOrArray, eps: number, s?: StreamOrDevice): array;
  export function layerNorm(array: ScalarOrArray, weights: ScalarOrArray | null, bias: ScalarOrArray | null, eps: number, s?: StreamOrDevice): array;
  export function rope(array: ScalarOrArray, dims: number, traditional: boolean, base: number, scale: number, offset: number, s?: StreamOrDevice): array;
  export function scaledDotProductAttention(queries: ScalarOrArray, keys: ScalarOrArray, values: ScalarOrArray, scale: number, mask?: ScalarOrArray, s?: StreamOrDevice): array;
}
