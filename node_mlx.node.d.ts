declare module '*node_mlx.node' {
  // Device.
  type DeviceType = number;

  const cpu: DeviceType;
  const gpu: DeviceType;

  class Device {
    type: DeviceType;
  }

  type DeviceOrType = Device | DeviceType;

  function defaultDevice(): Device;
  function setDefaultDevice(device: DeviceOrType): void;

  // Stream.
  class Stream {
    device: Device;
  }

  function defaultStream(device: DeviceOrType): Stream;
  function setDefaultStream(stream: Stream): void;
  function newStream(device: DeviceOrType): Stream;
  function toStream(s: StreamOrDevice): Stream;
  function stream(s: StreamOrDevice): Disposable;
  function synchronize(s?: Stream): void;

  type StreamOrDevice = Stream | Device | DeviceType | undefined;

  // Dtype.
  class Dtype {
    size: number;
  }

  const bool: Dtype;
  const bool_: Dtype;
  const uint8: Dtype;
  const uint16: Dtype;
  const uint32: Dtype;
  const uint64: Dtype;
  const int8: Dtype;
  const int16: Dtype;
  const int32: Dtype;
  const int64: Dtype;
  const float16: Dtype;
  const float32: Dtype;
  const float64: Dtype;
  const bfloat16: Dtype;
  const complex64: Dtype;

  type DtypeCategory = number;

  const complexfloating: DtypeCategory
  const floating: DtypeCategory
  const inexact: DtypeCategory
  const signedinteger: DtypeCategory
  const unsignedinteger: DtypeCategory
  const integer: DtypeCategory
  const number: DtypeCategory
  const generic: DtypeCategory

  // Complex number.
  interface Complex {
    re: number;
    im?: number;
  }
  function Complex(real: number, imag?: number): Complex;

  // Index slice.
  interface Slice {
    start: number | null;
    stop: number | null;
    step: number | null;
  }
  function Slice(start?: number | null, stop?: number | null, step?: number | null): Slice;

  // Array helper types.
  type MultiDimensionalArray<T> = MultiDimensionalArray<T>[] | T;
  type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array |
                    Int32Array | Uint32Array | Float32Array | Float64Array;
  type Scalar = boolean | number | Complex;
  type ScalarOrArray = MultiDimensionalArray<Scalar | TypedArray | array>;
  type ArrayIndex = null | Slice | '...' | array | number;
  // Helper class to apply updates at specific indices.
  class ArrayAt {
    constructor(array: array, indices: ArrayIndex | ArrayIndex[]);

    add(value: ScalarOrArray): array;
    subtract(value: ScalarOrArray): array;
    multiply(value: ScalarOrArray): array;
    divide(value: ScalarOrArray): array;
    maximum(value: ScalarOrArray): array;
    minimum(value: ScalarOrArray): array;
  }

  // The finfo is a class in python but we define it as function to simplify the
  // usage.
  function finfo(dtype: Dtype): { dtype: Dtype, min: number, max: number };

  // Array.
  function array(value: ScalarOrArray, dtype?: Dtype): array;

  class array {
    constructor(value: ScalarOrArray, dtype?: Dtype);

    length: number;
    astype(dtype: Dtype, s?: StreamOrDevice): array;
    at(...index: ArrayIndex[]): ArrayAt;
    item(): Scalar;
    tolist(): MultiDimensionalArray<Scalar>;
    toTypedArray(): TypedArray;
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
    cummax(axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
    cummin(axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
    cumprod(axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
    cumsum(axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
    diag(k?: number, s?: StreamOrDevice): array;
    diagonal(offset?: number, axis1?: number, axis2?: number, s?: StreamOrDevice): array;
    exp(s?: StreamOrDevice): array;
    power(exponent: ScalarOrArray, s?: StreamOrDevice): array;
    flatten(startAxis?: number, endAxis?: number, s?: StreamOrDevice): array;
    log(s?: StreamOrDevice): array;
    log10(s?: StreamOrDevice): array;
    log1p(s?: StreamOrDevice): array;
    log2(s?: StreamOrDevice): array;
    logsumexp(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
    max(keepdims?: boolean, s?: StreamOrDevice): array;
    max(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
    mean(keepdims?: boolean, s?: StreamOrDevice): array;
    mean(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
    min(keepdims?: boolean, s?: StreamOrDevice): array;
    min(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
    moveaxis(source: number, destination: number, s?: StreamOrDevice): array;
    prod(s?: StreamOrDevice): array;
    reciprocal(s?: StreamOrDevice): array;
    reshape(...shape: (number | number[])[]): array;
    round(s?: StreamOrDevice): array;
    rsqrt(s?: StreamOrDevice): array;
    sin(s?: StreamOrDevice): array;
    split(indicesOrSections?: number | number[], s?: StreamOrDevice): array[];
    sqrt(s?: StreamOrDevice): array;
    square(s?: StreamOrDevice): array;
    squeeze(axis?: number | number[], s?: StreamOrDevice): array;
    std(indicesOrSections?: number | number[], keepdims?: boolean, ddof?: number, s?: StreamOrDevice): array;
    sum(keepdims?: boolean, s?: StreamOrDevice): array;
    sum(axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
    swapaxes(axis1: number, axis2: number, s?: StreamOrDevice): array;
    transpose(...axes: (number | number[])[]): array;
    variance(indicesOrSections?: number | number[], keepdims?: boolean, ddof?: number, s?: StreamOrDevice): array;
    view(dtype: Dtype, s?: StreamOrDevice): array;

    index(...index: ArrayIndex[]): array;
    indexPut_(index: ArrayIndex | ArrayIndex[], value: ScalarOrArray): array;
    [Symbol.iterator](): IterableIterator<array>;
  }

  // Ops.
  function abs(array: ScalarOrArray, s?: StreamOrDevice): array;
  function add(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function addmm(a: ScalarOrArray, b: ScalarOrArray, c: ScalarOrArray, alpha?: number, beta?: number,  s?: StreamOrDevice): array;
  function all(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function all(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function allclose(a: ScalarOrArray, b: ScalarOrArray, rtol?: number, atol?: number, equalNan?: boolean, s?: StreamOrDevice): boolean;
  function any(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function any(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function arange(start: number, stop?: number, step?: number, dtype?: Dtype, s?: StreamOrDevice): array;
  function arange(stop: number, dtype?: Dtype, s?: StreamOrDevice): array;
  function arccos(array: ScalarOrArray, s?: StreamOrDevice): array;
  function arccosh(array: ScalarOrArray, s?: StreamOrDevice): array;
  function arcsin(array: ScalarOrArray, s?: StreamOrDevice): array;
  function arcsinh(array: ScalarOrArray, s?: StreamOrDevice): array;
  function arctan(array: ScalarOrArray, s?: StreamOrDevice): array;
  function arctanh(array: ScalarOrArray, s?: StreamOrDevice): array;
  function argmax(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function argmax(array: ScalarOrArray, axis?: number, keepdims?: boolean, s?: StreamOrDevice): array;
  function argmin(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function argmin(array: ScalarOrArray, axis?: number, keepdims?: boolean, s?: StreamOrDevice): array;
  function argpartition(array: ScalarOrArray, kth: number, axis?: number, s?: StreamOrDevice): array;
  function argsort(array: ScalarOrArray, s?: StreamOrDevice): array;
  function arrayEqual(a: ScalarOrArray, b: ScalarOrArray, equalNan?: boolean, s?: StreamOrDevice): array;
  function asStrided(array: ScalarOrArray, shape?: number[], strides?: number[], offset?: number, s?: StreamOrDevice): array;
  function atleast1d(...arrays: array[]): array;
  function atleast2d(...arrays: array[]): array;
  function atleast3d(...arrays: array[]): array;
  function issubdtype(a: Dtype | DtypeCategory, b: Dtype | DtypeCategory): boolean;
  function bitwiseAnd(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function bitwiseInvert(array: ScalarOrArray, s?: StreamOrDevice): array;
  function bitwiseOr(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function bitwiseXor(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function broadcastArrays(arrays: array[], s?: StreamOrDevice): array[];
  function broadcastTo(array: ScalarOrArray, shape: number | number[], s?: StreamOrDevice): array;
  function blockMaskedMM(a: ScalarOrArray, b: ScalarOrArray, blockSize: number, maskOut?: ScalarOrArray, maskLhs?: ScalarOrArray, maskRhs?: ScalarOrArray, s?: StreamOrDevice): array;
  function gatherMM(a: ScalarOrArray, b: ScalarOrArray, indicesLhs?: ScalarOrArray, indicesRhs?: ScalarOrArray, sortedIndices?: boolean, s?: StreamOrDevice): array;
  function gatherQMM(x: ScalarOrArray, w: ScalarOrArray, scales: ScalarOrArray, biases: ScalarOrArray, indicesLhs: ScalarOrArray, indicesRhs: ScalarOrArray, transpose: boolean, groupSize: number, bits: number, sortedIndices: boolean, s?: StreamOrDevice): array;
  function ceil(array: ScalarOrArray, s?: StreamOrDevice): array;
  function clip(array: ScalarOrArray, min: ScalarOrArray, max: ScalarOrArray, s?: StreamOrDevice): array;
  function concatenate(arrays?: ScalarOrArray[], axis?: number, s?: StreamOrDevice): array;
  function concat(arrays?: ScalarOrArray[], axis?: number, s?: StreamOrDevice): array;
  function convolve(input: ScalarOrArray, weight: ScalarOrArray, mode?: string, s?: StreamOrDevice): array;
  function conv1d(input: ScalarOrArray, weight: ScalarOrArray, stride?: number, padding?: number, dilation?: number, groups?: number, s?: StreamOrDevice): array;
  function conv2d(input: ScalarOrArray, weight: ScalarOrArray, stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, s?: StreamOrDevice): array;
  function conv3d(input: ScalarOrArray, weight: ScalarOrArray, stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, s?: StreamOrDevice): array;
  function convTranspose1d(input: ScalarOrArray, weight: ScalarOrArray, stride?: number, padding?: number, dilation?: number, groups?: number, s?: StreamOrDevice): array;
  function convTranspose2d(input: ScalarOrArray, weight: ScalarOrArray, stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, s?: StreamOrDevice): array;
  function convTranspose3d(input: ScalarOrArray, weight: ScalarOrArray, stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, s?: StreamOrDevice): array;
  function convGeneral(input: ScalarOrArray, weight?: ScalarOrArray, stride?: number | number[], padding?: number | number[] | [number[], number[]], kernelDilation?: number | number[], inputDilation?: number | number[], groups?: number, flip?: boolean, s?: StreamOrDevice): array;
  function cos(array: ScalarOrArray, s?: StreamOrDevice): array;
  function cosh(array: ScalarOrArray, s?: StreamOrDevice): array;
  function cummax(array: ScalarOrArray, axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
  function cummin(array: ScalarOrArray, axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
  function cumprod(array: ScalarOrArray, axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
  function cumsum(array: ScalarOrArray, axis?: number, reverse?: boolean, inclusive?: boolean, s?: StreamOrDevice): array;
  function degrees(array: ScalarOrArray, s?: StreamOrDevice): array;
  function dequantize(w: array, scales: ScalarOrArray, biases: ScalarOrArray, groupSize: number, bits: number, s?: StreamOrDevice): array;
  function diag(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
  function diagonal(array: ScalarOrArray, offset?: number, axis1?: number, axis2?: number, s?: StreamOrDevice): array;
  function divide(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function divmod(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): [array, array];
  function einsumPath(subscripts: string | string[], operands: array | array[], s?: StreamOrDevice): [[number, number][], string];
  function einsum(subscripts: string | string[], operands: array | array[], k: array, s?: StreamOrDevice): array;
  function equal(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function notEqual(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function erf(array: ScalarOrArray, s?: StreamOrDevice): array;
  function erfinv(array: ScalarOrArray, s?: StreamOrDevice): array;
  function exp(array: ScalarOrArray, s?: StreamOrDevice): array;
  function expm1(array: ScalarOrArray, s?: StreamOrDevice): array;
  function expandDims(array: ScalarOrArray, dims: number | number[], s?: StreamOrDevice): array;
  function eye(n: number, m?: number, k?: number, dtype?: Dtype, s?: StreamOrDevice): array;
  function flatten(array: ScalarOrArray, startAxis?: number, endAxis?: number, s?: StreamOrDevice): array;
  function floor(array: ScalarOrArray, s?: StreamOrDevice): array;
  function floorDivide(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function full(shape: number | number[], fillValue: ScalarOrArray, dtype?: Dtype, s?: StreamOrDevice): array;
  function greater(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function greaterEqual(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function identity(n: number, dtype?: Dtype, s?: StreamOrDevice): array;
  function imag(array: ScalarOrArray, s?: StreamOrDevice): array;
  function inner(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function isclose(a: ScalarOrArray, b: ScalarOrArray, rtol?: number, atol?: number, equalNan?: boolean, s?: StreamOrDevice): array;
  function isinf(array: ScalarOrArray, s?: StreamOrDevice): array;
  function isfinite(array: ScalarOrArray, s?: StreamOrDevice): array;
  function isnan(array: ScalarOrArray, s?: StreamOrDevice): array;
  function isneginf(array: ScalarOrArray, s?: StreamOrDevice): array;
  function isposinf(array: ScalarOrArray, s?: StreamOrDevice): array;
  function kron(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function leftShift(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function less(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function lessEqual(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function linspace(start: number, stop: number, num?: number, dtype?: Dtype, s?: StreamOrDevice): array;
  function load(filepath: string, s?: StreamOrDevice): array;
  function log(array: ScalarOrArray, s?: StreamOrDevice): array;
  function log2(array: ScalarOrArray, s?: StreamOrDevice): array;
  function log10(array: ScalarOrArray, s?: StreamOrDevice): array;
  function log1p(array: ScalarOrArray, s?: StreamOrDevice): array;
  function logaddexp(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function logicalNot(array: ScalarOrArray, s?: StreamOrDevice): array;
  function logicalAnd(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function logicalOr(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function logsumexp(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function matmul(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function max(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function max(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function maximum(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function mean(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function mean(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function meshgrid(...arrays: array[]): array[];
  function min(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function min(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function minimum(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function moveaxis(array: ScalarOrArray, source: number | number[], destination: number | number[], s?: StreamOrDevice): array;
  function multiply(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function nanToNum(array: array, nan: number, posinf?: number, neginf?: number, s?: StreamOrDevice): array;
  function negative(array: ScalarOrArray, s?: StreamOrDevice): array;
  function ones(shape: number | number[], dtype?: Dtype, s?: StreamOrDevice): array;
  function onesLike(array: ScalarOrArray, s?: StreamOrDevice): array;
  function outer(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function partition(array: ScalarOrArray, kth: number, axis?: number, s?: StreamOrDevice): array;
  function pad(array: ScalarOrArray, padWidth: number | [number] | [number, number] | [number, number][], constantValue?: ScalarOrArray, mode?: string, s?: StreamOrDevice): array;
  function power(array: ScalarOrArray, exponent: ScalarOrArray, s?: StreamOrDevice): array;
  function prod(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function prod(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function putAlongAxis(array: ScalarOrArray, indices: ScalarOrArray, values: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
  function quantize(w: array, groupSize: number, bits: number, s?: StreamOrDevice): array;
  function quantizedMatmul(w: array, x: array, scales: ScalarOrArray, biases: ScalarOrArray, transpose: boolean, groupSize: number, bits: number, s?: StreamOrDevice): array;
  function radians(array: ScalarOrArray, s?: StreamOrDevice): array;
  function real(array: ScalarOrArray, s?: StreamOrDevice): array;
  function reciprocal(array: ScalarOrArray, s?: StreamOrDevice): array;
  function remainder(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function repeat(array: ScalarOrArray, repeats?: number, axis?: number, s?: StreamOrDevice): array;
  function reshape(array: ScalarOrArray, ...shape: (number | number[])[]): array;
  function rightShift(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function roll(array: ScalarOrArray, shift?: number | number[], axis?: number | number[], s?: StreamOrDevice): array;
  function round(array: ScalarOrArray, s?: StreamOrDevice): array;
  function rsqrt(array: ScalarOrArray, s?: StreamOrDevice): array;
  function save(filepath: string, array: array): void;
  function saveGguf(filepath: string, arrays: Record<string, array>, metadata?: Record<string, string>): void;
  function saveSafetensors(filepath: string, arrays: Record<string, array>, metadata?: Record<string, string>): void;
  function sigmoid(array: ScalarOrArray, s?: StreamOrDevice): array;
  function sign(array: ScalarOrArray, s?: StreamOrDevice): array;
  function sin(array: ScalarOrArray, s?: StreamOrDevice): array;
  function sinh(array: ScalarOrArray, s?: StreamOrDevice): array;
  function slice(array: ScalarOrArray, startIndices: ScalarOrArray, axes: number[], sliceSize: number[], s?: StreamOrDevice): array;
  function sliceUpdate(src: ScalarOrArray, update: ScalarOrArray, startIndices: ScalarOrArray, axes: number[], s?: StreamOrDevice): array;
  function softmax(array: ScalarOrArray, axis?: number | number[], precise?: boolean, s?: StreamOrDevice): array;
  function sort(array: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
  function split(array: ScalarOrArray, indicesOrSections?: number | number[], axis?: number, s?: StreamOrDevice): array[];
  function sqrt(array: ScalarOrArray, s?: StreamOrDevice): array;
  function square(array: ScalarOrArray, s?: StreamOrDevice): array;
  function squeeze(array: ScalarOrArray, axis?: number | number[], s?: StreamOrDevice): array;
  function stack(arrays?: ScalarOrArray[], axis?: number, s?: StreamOrDevice): array;
  function std(array: ScalarOrArray, indicesOrSections?: number | number[], keepdims?: boolean, ddof?: number, s?: StreamOrDevice): array;
  function stopGradient(array: ScalarOrArray, s?: StreamOrDevice): array;
  function subtract(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
  function sum(array: ScalarOrArray, keepdims?: boolean, s?: StreamOrDevice): array;
  function sum(array: ScalarOrArray, axis?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
  function swapaxes(array: ScalarOrArray, axis1: number, axis2: number, s?: StreamOrDevice): array;
  function take(array: ScalarOrArray, indices?: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
  function takeAlongAxis(array: ScalarOrArray, indices?: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
  function tan(array: ScalarOrArray, s?: StreamOrDevice): array;
  function tanh(array: ScalarOrArray, s?: StreamOrDevice): array;
  function tensordot(a: ScalarOrArray, b: ScalarOrArray, axes?: number | [number | number[], number | number[]], s?: StreamOrDevice): array;
  function tile(array: ScalarOrArray, reps?: number[], s?: StreamOrDevice): array;
  function topk(array: ScalarOrArray, k: number, axis?: number, s?: StreamOrDevice): [array, array];
  function trace(array: ScalarOrArray, offset: number, axis1: number, axis2: number, dtype?: Dtype, s?: StreamOrDevice): [array, array];
  function transpose(array: ScalarOrArray, ...axes: (number | number[])[]): array;
  function permuteDims(array: ScalarOrArray, ...axes: (number | number[])[]): array;
  function tri(N: number, M?: number, k?: number, dtype?: Dtype, s?: StreamOrDevice): array;
  function tril(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
  function triu(array: ScalarOrArray, k?: number, s?: StreamOrDevice): array;
  function unflatten(array: ScalarOrArray, axis: number, shape: number[], s?: StreamOrDevice): array;
  function variance(array: ScalarOrArray, indicesOrSections?: number | number[], keepdims?: boolean, ddof?: number, s?: StreamOrDevice): array;
  function where(condition: ScalarOrArray, x: ScalarOrArray, y: ScalarOrArray, s?: StreamOrDevice): array;
  function zeros(shape: number | number[], dtype?: Dtype, s?: StreamOrDevice): array;
  function zerosLike(array: ScalarOrArray, s?: StreamOrDevice): array;

  // Transforms.
  // @ts-ignore: eval is a special function in JS and this fails type check.
  function eval(...args: unknown[]): void;
  function asyncEval(...args: unknown[]): Promise<void>;
  function jvp(func: (...args: array[]) => array | array[], primals: array[], tangents: array[]): [array[], array[]];
  function vjp(func: (...args: array[]) => array | array[], primals: array[], cotangents: array[]): [array[], array[]];
  function valueAndGrad<T extends any[], U>(func: (...args: T) => U, argnums?: number | number[]): (...args: T) => [U, array];
  function grad<T extends any[], U>(func: (...args: T) => U, argnums?: number | number[]): (...args: T) => U;
  function vmap<T extends any[], U>(func: (...args: T) => U, inAxes?: number | number[], outAxes?: number | number[]): (...args: T) => U;
  function compile<T extends any[], U>(func: (...args: T) => U, shapeless?: boolean): (...args: T) => U;
  function disableCompile(): void;
  function enableCompile(): void;
  function checkpoint<T extends any[], U>(func: (...args: T) => U): (...args: T) => U;

  // Memory.
  function getActiveMemory(): number;
  function getPeakMemory(): number;
  function resetPeakMemory(): void;
  function getCacheMemory(): number;
  function setMemoryLimit(limit: number): number;
  function setWiredLimit(limit: number): number;
  function setCacheLimit(limit: number): number;
  function clearCache(): void;

  // JavaScript APIs.
  function tidy<U>(func: () => U): U;
  function dispose(...args: unknown[]): void;
  function getWrappersCount(): number;

  // Metal.
  namespace metal {
    interface DeviceInfo {
      architecture: string,
      max_buffer_length: number,
      max_recommended_working_set_size: number,
      memory_size: number,
    }

    function isAvailable(): boolean;
    function startCapture(path: string): boolean;
    function stopCapture(): void;
    function deviceInfo(): DeviceInfo;
  }

  // Random.
  namespace random {
    function bernoulli(p?: ScalarOrArray, shape?: number[], key?: array, s?: StreamOrDevice): array;
    function categorical(logits: ScalarOrArray, axis?: number, shape?: number[], numSamples?: number, key?: array, s?: StreamOrDevice): array;
    function gumbel(shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
    function key(seed: number): array;
    function normal(shape?: number[], dtype?: Dtype, loc?: number, scale?: number, key?: array, s?: StreamOrDevice): array;
    function multivariateNormal(mean: array, cov: array, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
    function randint(low: ScalarOrArray, high: ScalarOrArray, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
    function seed(seed: number): void;
    function split(array: array, num?: number, s?: StreamOrDevice): array;
    function truncatedNormal(lower: ScalarOrArray, upper: ScalarOrArray, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
    function uniform(low: ScalarOrArray, high: ScalarOrArray, shape?: number[], dtype?: Dtype, key?: array, s?: StreamOrDevice): array;
    function laplace(shape?: number[], dtype?: Dtype, loc?: number, scale?: number, key?: array, s?: StreamOrDevice): array;
    function permutation(x: number, key?: array, s?: StreamOrDevice): array;
    function permutation(x: array, axis?: number, key?: array, s?: StreamOrDevice): array;
  }

  // FFT.
  namespace fft {
    function fft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
    function ifft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
    function fft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function ifft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function fftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function ifftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function rfft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
    function irfft(array: ScalarOrArray, n?: number, axis?: number, s?: StreamOrDevice): array;
    function rfft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function irfft2(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function rfftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
    function irfftn(array: ScalarOrArray, n?: number[], axes?: number[], s?: StreamOrDevice): array;
  }

  // Linear algebra.
  namespace linalg {
    function norm(array: ScalarOrArray, norm?: number | string, axes?: number | number[], keepdims?: boolean, s?: StreamOrDevice): array;
    function inv(array: ScalarOrArray, s?: StreamOrDevice): array;
    function triInv(array: ScalarOrArray, upper: boolean, s?: StreamOrDevice): array;
    function cholesky(array: ScalarOrArray, upper: boolean, s?: StreamOrDevice): array;
    function choleskyInv(array: ScalarOrArray, upper: boolean, s?: StreamOrDevice): array;
    function pinv(array: ScalarOrArray, s?: StreamOrDevice): array;
    function cross(a: ScalarOrArray, b: ScalarOrArray, axis?: number, s?: StreamOrDevice): array;
    function qr(array: ScalarOrArray, s?: StreamOrDevice): array[];
    function svd(array: ScalarOrArray, s?: StreamOrDevice): array[];
    function svd(array: ScalarOrArray, computeUv: boolean, s?: StreamOrDevice): array[];
    function eigvalsh(array: ScalarOrArray, uplo?: string, s?: StreamOrDevice): array[];
    function eigh(array: ScalarOrArray, uplo?: string, s?: StreamOrDevice): array[];
    function lu(array: ScalarOrArray, s?: StreamOrDevice): array[];
    function luFactor(array: ScalarOrArray, s?: StreamOrDevice): [array, array];
    function solve(a: ScalarOrArray, b: ScalarOrArray, s?: StreamOrDevice): array;
    function solveTriangular(a: ScalarOrArray, b: ScalarOrArray, upper: boolean, s?: StreamOrDevice): array;
  }

  // Fast operations.
  namespace fast {
    function rmsNorm(array: ScalarOrArray, weights: ScalarOrArray, eps: number, s?: StreamOrDevice): array;
    function layerNorm(array: ScalarOrArray, weights: ScalarOrArray | null, bias: ScalarOrArray | null, eps: number, s?: StreamOrDevice): array;
    function rope(array: ScalarOrArray, dims: number, traditional: boolean, base: number | undefined, scale: number, offset: ScalarOrArray, freqs?: array, s?: StreamOrDevice): array;
    function scaledDotProductAttention(queries: ScalarOrArray, keys: ScalarOrArray, values: ScalarOrArray, scale: number, mask?: ScalarOrArray, memoryEfficientThreshold?: number, s?: StreamOrDevice): array;
    function affineQuantize(w: ScalarOrArray, scales: ScalarOrArray, biases: ScalarOrArray, groupSize?: number, bits?: number, s?: StreamOrDevice): array;
  }

  // Constants.
  const Inf: number;
  const Infinity: number;
  const inf: number;
  const infty: number;
  const NAN: number;
  const NaN: number;
  const nan: number;
  const NINF: number;
  const NZERO: number;
  const PINF: number;
  const PZERO: number;
  const e: number;
  const eulerGamma: number;
  const pi: number;
  const newaxis: null;
}

