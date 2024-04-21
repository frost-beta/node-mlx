export type DeviceType = number;

export const cpu: DeviceType;
export const gpu: DeviceType;

export class Device {
  type: DeviceType;
}

export function defaultDevice(): Device;
export function setDefaultDevice(device: Device);

export class Stream {
  device: Device;
}

export function defaultStream(device: Device): Stream;
export function setDefaultStream();
export function newStream(device: Device): Stream;

type StreamOrDevice = Stream | Device | DeviceType | undefined;

export class Dtype {
  size: number;
}

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

type MultiDimensionalArray<T> = MultiDimensionalArray<T>[] | T;

export function array(value: MultiDimensionalArray<boolean | number>, dtype: Dtype? = 'float32'): array;

export class array {
  constructor(value: MultiDimensionalArray<boolean | number>, dtype: Dtype? = 'float32');

  length: number;
  astype(dtype: Dtype, s?: StreamOrDevice): array;
  at(index: number, s?: StreamOrDevice): array;
  item(): boolean | number;
  tolist(): MultiDimensionalArray<boolean | number>;
  dtype: Dtype;
  itemsize: number;
  nbytes?: number;
  ndim: number;
  shape: number[];
  size: number;
  abs(s?: StreamOrDevice): array;
  all(s?: StreamOrDevice): array;
  any(s?: StreamOrDevice): array;
  argmax(s?: StreamOrDevice): array;
  argmin(s?: StreamOrDevice): array;
  cos(s?: StreamOrDevice): array;
  cummax(s?: StreamOrDevice): array;
  cummin(s?: StreamOrDevice): array;
  cumprod(s?: StreamOrDevice): array;
  cumsum(s?: StreamOrDevice): array;
  diag(s?: StreamOrDevice): array;
  diagonal(s?: StreamOrDevice): array;
  exp(s?: StreamOrDevice): array;
  flatten(s?: StreamOrDevice): array;
  log(s?: StreamOrDevice): array;
  log10(s?: StreamOrDevice): array;
  log1p(s?: StreamOrDevice): array;
  log2(s?: StreamOrDevice): array;
  logsumexp(s?: StreamOrDevice): array;
  max(s?: StreamOrDevice): array;
  mean(s?: StreamOrDevice): array;
  min(s?: StreamOrDevice): array;
  moveaxis(source: number, destination: number, s?: StreamOrDevice): array;
  prod(s?: StreamOrDevice): array;
  reciprocal(s?: StreamOrDevice): array;
  reshape(shape: number[], s?: StreamOrDevice): array;
  round(s?: StreamOrDevice): array;
  rsqrt(s?: StreamOrDevice): array;
  sin(s?: StreamOrDevice): array;
  split(indicesOrSections?: number | number[], s?: StreamOrDevice): array[];
  sqrt(s?: StreamOrDevice): array;
  square(s?: StreamOrDevice): array;
  squeeze(s?: StreamOrDevice): array;
  swapaxes(axis1: number, axis2: number, s?: StreamOrDevice): array;
  sum(s?: StreamOrDevice): array;
  transpose(s?: StreamOrDevice): array;
  T: array;
  var(s?: StreamOrDevice): array;
}

type ScalarOrArray = boolean | number | number[] | array;

export function abs(array: ScalarOrArray, s?: StreamOrDevice): array;
export function add(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function all(array: ScalarOrArray, s?: StreamOrDevice): array;
export function allclose(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): boolean;
export function any(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arange(start: number, stop: number, step: number, dtype: Dtype, s?: StreamOrDevice): array;
export function arccos(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arccosh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arcsin(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arcsinh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arctan(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arctanh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function argmax(array: ScalarOrArray, s?: StreamOrDevice): array;
export function argmin(array: ScalarOrArray, s?: StreamOrDevice): array;
export function argpartition(array: ScalarOrArray, kth: number, s?: StreamOrDevice): array;
export function argsort(array: ScalarOrArray, s?: StreamOrDevice): array;
export function arrayEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): boolean;
export function atleast1d(...arrays?: array[]): array;
export function atleast2d(...arrays?: array[]): array;
export function atleast3d(...arrays?: array[]): array;
export function broadcastTo(array: ScalarOrArray, shape: number[], s?: StreamOrDevice): array;
export function ceil(array: ScalarOrArray, s?: StreamOrDevice): array;
export function clip(array: ScalarOrArray, aMin: number, aMax: number, s?: StreamOrDevice): array;
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
export function diag(array: ScalarOrArray, s?: StreamOrDevice): array;
export function diagonal(array: ScalarOrArray, s?: StreamOrDevice): array;
export function divide(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function divmod(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): [array, array];
export function equal(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function notEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function erf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function erfinv(array: ScalarOrArray, s?: StreamOrDevice): array;
export function exp(array: ScalarOrArray, s?: StreamOrDevice): array;
export function expm1(array: ScalarOrArray, s?: StreamOrDevice): array;
export function expandDims(array: ScalarOrArray, axis?: number | null, s?: StreamOrDevice): array;
export function eye(N: number, M?: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function flatten(array: ScalarOrArray, s?: StreamOrDevice): array;
export function floor(array: ScalarOrArray, s?: StreamOrDevice): array;
export function floorDivide(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function full(shape: number[], fillValue: any, dtype?: Dtype, s?: StreamOrDevice): array;
export function greater(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function greaterEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function identity(n: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function inner(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function isclose(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function isinf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function isnan(array: ScalarOrArray, s?: StreamOrDevice): array;
export function isneginf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function isposinf(array: ScalarOrArray, s?: StreamOrDevice): array;
export function less(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function lessEqual(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function linspace(start: number, stop: number, num: number, dtype?: Dtype, s?: StreamOrDevice): array;
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
export function max(array: ScalarOrArray, s?: StreamOrDevice): array;
export function maximum(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function mean(array: ScalarOrArray, s?: StreamOrDevice): array;
export function meshgrid(...arrays?: array[]): array[];
export function min(array: ScalarOrArray, s?: StreamOrDevice): array;
export function minimum(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function moveaxis(array: ScalarOrArray, source: number | number[], destination: number | number[], s?: StreamOrDevice): array;
export function multiply(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function negative(array: ScalarOrArray, s?: StreamOrDevice): array;
export function ones(shape: number[], dtype?: Dtype, s?: StreamOrDevice): array;
export function onesLike(array: ScalarOrArray, s?: StreamOrDevice): array;
export function outer(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function partition(array: ScalarOrArray, kth: number | number[], axis?: number, s?: StreamOrDevice): array;
export function pad(array: ScalarOrArray, padWidth: number[][], mode?: string, constantValues?: any, s?: StreamOrDevice): array;
export function prod(array: ScalarOrArray, s?: StreamOrDevice): array;
export function quantize(array: ScalarOrArray, s?: StreamOrDevice): array;
export function quantizedMatmul(array1: ScalarOrArray, array2: ScalarOrArray, s?: StreamOrDevice): array;
export function reciprocal(array: ScalarOrArray, s?: StreamOrDevice): array;
export function repeat(array: ScalarOrArray, repeats?: number | number[], axis?: number, s?: StreamOrDevice): array;
export function reshape(array: ScalarOrArray, shape: number[], s?: StreamOrDevice): array;
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
export function sum(array: ScalarOrArray, s?: StreamOrDevice): array;
export function swapaxes(array: ScalarOrArray, axis1: number, axis2: number, s?: StreamOrDevice): array;
export function take(array: ScalarOrArray, indices?: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
export function takeAlongAxis(array: ScalarOrArray, indices?: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
export function tan(array: ScalarOrArray, s?: StreamOrDevice): array;
export function tanh(array: ScalarOrArray, s?: StreamOrDevice): array;
export function tensordot(array1: ScalarOrArray, array2: ScalarOrArray, axes?: number | [number | number[], number | number[]], s?: StreamOrDevice): array;
export function tile(array: ScalarOrArray, reps?: number[], s?: StreamOrDevice): array;
export function topk(array: ScalarOrArray, k: number, s?: StreamOrDevice): [array, array];
export function transpose(array: ScalarOrArray, axes?: number[], s?: StreamOrDevice): array;
export function tri(N: number, M?: number, k?: number, dtype?: Dtype, s?: StreamOrDevice): array;
export function tril(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
export function triu(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
export function variance(array: ScalarOrArray, s?: StreamOrDevice): array;
export function where(condition: ScalarOrArray, x: array | null, y: array | null, s?: StreamOrDevice): array;
export function zeros(shape: number[], dtype?: Dtype, s?: StreamOrDevice): array;
export function zerosLike(array: ScalarOrArray, s?: StreamOrDevice): array;
