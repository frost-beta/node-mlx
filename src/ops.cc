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
          "takeAlongAxis", &ops::TakeAlongAxis
          );
}
