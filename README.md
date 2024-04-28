# node-mlx

:construction:

## Usage

```typescript
import mx from '@frost-beta/mlx';

const a = mx.arange(64).reshape([8, 8]);
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

### Unimplemented features

Some features are not supported yet and will be implemented in future:

* The function passed to `mx.grad`/`mx.valueAndGrad`/`mx.vmap`/`mx.compile` must
  have all its parameters taking `mx.array`.
* When creating a `mx.array` from JavaScript Array, the Array must only include
  primitive values.

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
