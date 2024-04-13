#include "src/array.h"
#include "src/ops.h"

namespace ops {

mx::array Flatten(const mx::array& a,
                  std::optional<int> start_axis,
                  std::optional<int> end_axis,
                  mx::StreamOrDevice s) {
  return mx::flatten(a, start_axis.value_or(0), end_axis.value_or(-1), s);
}

mx::array Squeeze(const mx::array& a,
                  IntOrVector axis,
                  mx::StreamOrDevice s) {
  if (std::holds_alternative<std::monostate>(axis)) {
    return squeeze(a, s);
  } else if (auto i = std::get_if<int>(&axis); i) {
    return squeeze(a, *i, s);
  } else {
    return squeeze(a, std::move(std::get<std::vector<int>>(axis)), s);
  }
}

mx::array ExpandDims(const mx::array& a,
                     std::variant<int, std::vector<int>> dims,
                     mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&dims); i) {
    return expand_dims(a, *i, s);
  } else {
    return expand_dims(a, std::move(std::get<std::vector<int>>(dims)), s);
  }
}

mx::array ArrayEqual(const mx::array& a,
                     const mx::array& b,
                     std::optional<bool> equal_nan,
                     mx::StreamOrDevice s) {
  return mx::array_equal(a, b, equal_nan.value_or(false), s);
}

mx::array ARange(float first, ki::Arguments* args) {
  // The first arg is |stop| if arange is called with only 1 number.
  bool first_arg_is_start = false;
  if (args->Length() > 0)
    first_arg_is_start = ki::IsType(args->Env(), (*args)[1], napi_number);
  // Manual parsing args.
  float start = first_arg_is_start ? first : 0;
  float stop = first_arg_is_start ? args->GetNext<float>().value() : first;
  float step = args->TryGetNext<float>().value_or(1);
  auto dtype = args->TryGetNext<mx::Dtype>().value_or(mx::float32);
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  return mx::arange(start, stop, step, dtype, s);
}

mx::array Linspace(float start,
                   float stop,
                   std::optional<int> num,
                   std::optional<mx::Dtype> dtype,
                   mx::StreamOrDevice s) {
  return mx::linspace(start, stop, num.value_or(50),
                      dtype.value_or(mx::float32), s);
}

mx::array Take(const mx::array& a,
               const mx::array& indices,
               std::optional<int> axis,
               mx::StreamOrDevice s) {
  if (axis)
    return mx::take(a, indices, *axis, s);
  else
    return mx::take(a, indices, s);
}

mx::array TakeAlongAxis(const mx::array& a,
                        const mx::array& indices,
                        std::optional<int> axis,
                        mx::StreamOrDevice s) {
  if (axis)
    return mx::take_along_axis(a, indices, *axis, s);
  else
    return mx::take_along_axis(reshape(a, {-1}, s), indices, 0, s);
}

mx::array Full(std::variant<int, std::vector<int>> shape,
               ScalarOrArray vals,
               std::optional<mx::Dtype> dtype,
               mx::StreamOrDevice s) {
  return mx::full(ToShape(std::move(shape)),
                  ToArray(std::move(vals), std::move(dtype)),
                  s);
}

mx::array Zeros(std::variant<int, std::vector<int>> shape,
                std::optional<mx::Dtype> dtype,
                mx::StreamOrDevice s) {
  return mx::zeros(ToShape(std::move(shape)), dtype.value_or(mx::float32), s);
}

mx::array Ones(std::variant<int, std::vector<int>> shape,
               std::optional<mx::Dtype> dtype,
               mx::StreamOrDevice s) {
  return mx::ones(ToShape(std::move(shape)), dtype.value_or(mx::float32), s);
}

mx::array Eye(int n,
              std::optional<int> m,
              std::optional<int> k,
              std::optional<mx::Dtype> dtype,
              mx::StreamOrDevice s) {
  return mx::eye(n, m.value_or(n), k.value_or(0), dtype.value_or(mx::float32),
                 s);
}

mx::array Identity(int n,
                   std::optional<mx::Dtype> dtype,
                   mx::StreamOrDevice s) {
  return mx::identity(n, dtype.value_or(mx::float32), s);
}

mx::array Tri(int n,
              std::optional<int> m,
              std::optional<int> k,
              std::optional<mx::Dtype> dtype,
              mx::StreamOrDevice s) {
  return mx::tri(n, m.value_or(n), k.value_or(0), dtype.value_or(mx::float32),
                 s);
}

mx::array Transpose(const mx::array& a,
                    std::optional<std::vector<int>> axis,
                    mx::StreamOrDevice s) {
  if (axis)
    return mx::transpose(a, GetReduceAxes(std::move(*axis), a.ndim()), s);
  else
    return mx::transpose(a);
}

mx::array Var(const mx::array& a,
              IntOrVector axis,
              std::optional<bool> keepdims,
              std::optional<int> ddof,
              mx::StreamOrDevice s) {
  return mx::var(a, GetReduceAxes(std::move(axis), a.ndim()),
                 keepdims.value_or(false), ddof.value_or(0), s);
}

mx::array Std(const mx::array& a,
              IntOrVector axis,
              std::optional<bool> keepdims,
              std::optional<int> ddof,
              mx::StreamOrDevice s) {
  return mx::std(a, GetReduceAxes(std::move(axis), a.ndim()),
                 keepdims.value_or(false), ddof.value_or(0), s);
}

std::vector<mx::array> Split(const mx::array& a,
                             std::variant<int, std::vector<int>> indices,
                             std::optional<int> axis,
                             mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&indices); i) {
    return mx::split(a, *i, axis.value_or(0), s);
  } else {
    return mx::split(a, std::move(std::get<std::vector<int>>(indices)),
                     axis.value_or(0), s);
  }
}

mx::array ArgMin(const mx::array& a,
                 std::optional<int> axis,
                 std::optional<bool> keepdims,
                 mx::StreamOrDevice s) {
  if (axis)
    return mx::argmin(a, *axis, keepdims.value_or(false), s);
  else
    return mx::argmin(a, keepdims.value_or(false), s);
}

mx::array ArgMax(const mx::array& a,
                 std::optional<int> axis,
                 std::optional<bool> keepdims,
                 mx::StreamOrDevice s) {
  if (axis)
    return mx::argmax(a, *axis, keepdims.value_or(false), s);
  else
    return mx::argmax(a, keepdims.value_or(false), s);
}

mx::array Sort(const mx::array& a,
               std::optional<int> axis,
               mx::StreamOrDevice s) {
  if (axis)
    return mx::sort(a, *axis, s);
  else
    return mx::sort(a, s);
}

mx::array ArgSort(const mx::array& a,
                  std::optional<int> axis,
                  mx::StreamOrDevice s) {
  if (axis)
    return mx::argsort(a, *axis, s);
  else
    return mx::argsort(a, s);
}

mx::array Partition(const mx::array& a,
                    int kth,
                    std::optional<int> axis,
                    mx::StreamOrDevice s) {
  if (axis)
    return mx::partition(a, kth, *axis, s);
  else
    return mx::partition(a, kth, s);
}

mx::array ArgPartition(const mx::array& a,
                       int kth,
                       std::optional<int> axis,
                       mx::StreamOrDevice s) {
  if (axis)
    return mx::argpartition(a, kth, *axis, s);
  else
    return mx::argpartition(a, kth, s);
}

mx::array TopK(const mx::array& a,
               int k,
               std::optional<int> axis,
               mx::StreamOrDevice s) {
  if (axis)
    return mx::topk(a, k, *axis, s);
  else
    return mx::topk(a, k, s);
}

mx::array Softmax(const mx::array& a,
                  IntOrVector axis,
                  bool precise,
                  mx::StreamOrDevice s) {
  return mx::softmax(a, GetReduceAxes(std::move(axis), a.ndim()), precise, s);
}

mx::array Concatenate(std::vector<mx::array> arrays,
                      std::optional<int> axis,
                      mx::StreamOrDevice s) {
  if (axis)
    return mx::concatenate(std::move(arrays), *axis, s);
  else
    return mx::concatenate(std::move(arrays), s);
}

mx::array Stack(std::vector<mx::array> arrays,
                std::optional<int> axis,
                mx::StreamOrDevice s) {
  if (axis)
    return mx::stack(std::move(arrays), *axis, s);
  else
    return mx::stack(std::move(arrays), s);
}

mx::array Repeat(const mx::array& a,
                 int repeats,
                 std::optional<int> axis,
                 mx::StreamOrDevice s) {
  if (axis)
    return mx::repeat(a, repeats, *axis, s);
  else
    return mx::repeat(a, repeats, s);
}

}  // namespace ops

void InitOps(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "reshape", &mx::reshape,
          "flatten", &ops::Flatten,
          "squeeze", &ops::Squeeze,
          "expandDims", &ops::ExpandDims,
          "abs", &mx::abs,
          "sign", &mx::sign,
          "negative", &mx::negative,
          "add", &mx::add,
          "subtract", &mx::subtract,
          "multiply", &mx::multiply,
          "divide", &mx::divide,
          "divmod", &mx::divmod,
          "floorDivide", &mx::floor_divide,
          "remainder", &mx::remainder,
          "equal", &mx::equal,
          "notEqual", &mx::not_equal,
          "less", &mx::less,
          "lessEqual", &mx::less_equal,
          "greater", &mx::greater,
          "greaterEqual", &mx::greater_equal,
          "arrayEqual", &ops::ArrayEqual,
          "matmul", &mx::matmul,
          "square", &mx::square,
          "sqrt", &mx::sqrt,
          "rsqrt", &mx::rsqrt,
          "reciprocal", &mx::reciprocal,
          "logicalNot", &mx::logical_not,
          "logicalAnd", &mx::logical_and,
          "logicalOr", &mx::logical_or,
          "logaddexp", &mx::logaddexp,
          "exp", &mx::exp,
          "expm1", &mx::expm1,
          "erf", &mx::erf,
          "erfinv", &mx::erfinv,
          "sin", &mx::sin,
          "cos", &mx::cos,
          "tan", &mx::tan,
          "arcsin", &mx::arcsin,
          "arccos", &mx::arccos,
          "arctan", &mx::arctan,
          "sinh", &mx::sinh,
          "cosh", &mx::cosh,
          "tanh", &mx::tanh,
          "arcsinh", &mx::arcsinh,
          "arccosh", &mx::arccosh,
          "arctanh", &mx::arctanh,
          "log", &mx::log,
          "log2", &mx::log2,
          "log10", &mx::log10,
          "log1p", &mx::log1p,
          "stopGradient", &mx::stop_gradient,
          "sigmoid", &mx::sigmoid,
          "power", &mx::power,
          "arange", &ops::ARange,
          "linspace", &ops::Linspace,
          "take", &ops::Take,
          "takeAlongAxis", &ops::TakeAlongAxis,
          "full", &ops::Full,
          "zeros", &ops::Zeros,
          "zerosLike", &mx::zeros_like,
          "ones", &ops::Ones,
          "onesLike", &mx::ones_like,
          "eye", &ops::Eye,
          "identity", &ops::Identity,
          "tri", &ops::Tri,
          "tril", &mx::tril,
          "allclose", &mx::allclose,
          "isclose", &mx::isclose,
          "all", DimOpWrapper(&mx::all),
          "any", DimOpWrapper(&mx::any),
          "minimum", &mx::minimum,
          "maximum", &mx::maximum,
          "floor", &mx::floor,
          "ceil", &mx::ceil,
          "isnan", &mx::isnan,
          "isinf", &mx::isinf,
          "isposinf", &mx::isposinf,
          "isneginf", &mx::isneginf,
          "moveaxis", &mx::moveaxis,
          "swapaxes", &mx::swapaxes,
          "transpose", &ops::Transpose,
          "sum", DimOpWrapper(&mx::sum),
          "prod", DimOpWrapper(&mx::prod),
          "min", DimOpWrapper(&mx::min),
          "max", DimOpWrapper(&mx::max),
          "logsumexp", DimOpWrapper(&mx::logsumexp),
          "mean", DimOpWrapper(&mx::mean),
          "var", &ops::Var,
          "std", &ops::Std,
          "split", &ops::Split,
          "argmin", &ops::ArgMin,
          "argmax", &ops::ArgMax,
          "sort", &ops::Sort,
          "argsort", &ops::ArgSort,
          "partition", &ops::Partition,
          "argpartition", &ops::ArgPartition,
          "topk", &ops::TopK,
          "broadcastTo", &mx::broadcast_to,
          "softmax", &ops::Softmax,
          "concatenate", &ops::Concatenate,
          "stack", &ops::Stack,
          "meshgrid", &mx::meshgrid,
          "repeat", &ops::Repeat,
          "clip", &mx::clip);
}
