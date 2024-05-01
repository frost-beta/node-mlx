# node-mlx

:construction:

This project is not affiliated to Apple, you can sponsorits development by
[sponsoring me](https://github.com/sponsors/zcbenz).

## Supported platforms

GPU support:
- Macs with Apple Silicon

CPU support:
- x64 Macs
- x64/arm64 Linux

(No Windows support but I'll try to make MLX work on it in future.)

Note that currently MLX does not have plans to support GPUs other than Apple
Silicon, and personally I don't think they ever will considering their team size
and the API design.

For doing machine learning on GPUs with Node.js, you can go with TensorFlow.js,
or wait for someone porting PyTorch to Node.js (could be me, haha).

## Usage

```typescript
import {core as mx, nn} from '@frost-beta/mlx';

const a = mx.arange(64).reshape(8, 8);
console.log(a.shape);
```

## APIs

There is currently no documentations for JavaScript APIs, please check the
TypeScript definitions for available APIs, and [MLX's official website](https://ml-explore.github.io/mlx/)
for documentations.

The JavaScript APIs basically duplicate the official Python APIs by converting
the API names from snake_case to camcelCase. For example the `mx.not_equal`
Python API is renamed to `mx.notEqual` in JavaScript.

There are a few exceptions due to limitations of JavaScript:

* JavaScript numbers are always floating-point values, so the default dtype
  of `mx.array(42)` is `mx.float32` instead of `mx.int32`.
* The `mx.var` API is renamed to `mx.variance`.
* Operator overloading does not work, use `mx.add(a, b)` instead of `a + b`.
* Indexing via `[]` operator does not work, use `array.item` and
  `array.itemPut_` methods instead (the `_` suffix means inplace operation).
* `delete array` does nothing, you must wait for garbage collection to get the
  array's memory freed.
* The `Module` instances can not be used as functions, the `forward` method must
  be used instead.

### Unimplemented features

Some features are not supported yet and will be implemented in future:

* The function passed to `mx.vmap`/`mx.compile` must have all its parameters
  taking `mx.array`.
* When creating a `mx.array` from JavaScript Array, the Array must only include
  primitive values.
* The APIs only accept plain parameters, e.g. `mx.uniform(0, 1, [2, 2])`. Named
  parameter calls like `mx.uniform({shape: [2, 2]})` has not been implemented.

### Complex numbers

There is no built-in complex numbers in JavaScript, and we use objects to
represent them:

```typescript
interface Complex {
  re: number;
  im: number;
}
```

You can also use the `mx.Complex(real, imag?)` helper to create complex numbers.

### Indexing

Slice in JavaScript is represented as object:

```typescript
interface Slice {
  start: number | null;
  stop: number | null;
  step: number | null;
}
```

You can also use the `mx.Slice(start?, stop?, step?)` helper to create slices.

The JavaScript standard does not allow using `...` as values. To use ellipsis as
index, use string `"..."` instead.

When using arrays as indices, make sure a integer dtype is specified because
the default dtype is `float32`, for example
`a.index(mx.array([ 1, 2, 3 ], mx.uint32))`.

Here are some examples of translating Python indexing code to JavaScript:

#### Getters

#### Setters

#### Translating between Python/JavaScript index types

## Versioning

Before matching the features and stability of the official Python APIs, this
project will stay on 0.0.x for versions.
