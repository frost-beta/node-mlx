#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <numeric>

#include <mlx/mlx.h>
#include <kizunapi.h>

namespace mx = mlx::core;

using IntOrVector = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::variant<bool, float, mx::array>;

// In js land the objects are always stored as pointers, when a value is needed
// from C++ land, we do a copy.
template<typename T>
inline std::optional<T> NodeObjToCppValue(napi_env env, napi_value value) {
  std::optional<T*> ptr = ki::FromNode<T*>(env, value);
  if (!ptr)
    return std::nullopt;
  return *ptr.value();
}

// Get axis arg from js value.
inline std::vector<int> GetReduceAxes(IntOrVector value, int dims) {
  // Try vector first.
  if (auto v = std::get_if<std::vector<int>>(&value); v)
    return std::move(*v);
  // Then int.
  if (auto i = std::get_if<int>(&value); i)
    return {*i};
  // Default to all dims.
  std::vector<int> all(dims);
  std::iota(all.begin(), all.end(), 0);
  return all;
}

// Convert a int or vector into shape.
inline std::vector<int> ToShape(std::variant<int, std::vector<int>> shape) {
  if (auto i = std::get_if<int>(&shape); i)
    return {*i};
  return std::move(std::get<std::vector<int>>(shape));
}

// Convert a ScalarOrArray arg to array.
inline mx::array ToArray(ScalarOrArray value,
                         std::optional<mx::Dtype> dtype = std::nullopt) {
  if (auto a = std::get_if<mx::array>(&value); a)
    return std::move(*a);
  if (auto b = std::get_if<bool>(&value); b)
    return mx::array(*b, dtype.value_or(mx::bool_));
  if (auto f = std::get_if<float>(&value); f) {
    mx::Dtype out_dtype = dtype.value_or(mx::float32);
    return mx::array(*f, mx::issubdtype(out_dtype, mx::floating) ? out_dtype
                                                                 : mx::float32);
  }
  throw std::invalid_argument("Invalid type passed to ToArray");
}

// A template converter for ops that accept |axis|.
inline
std::function<mx::array(const mx::array& a,
                        IntOrVector axis,
                        mx::StreamOrDevice s)>
DimOpWrapper(mx::array(*func)(const mx::array&,
                              const std::vector<int>&,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                IntOrVector axis,
                mx::StreamOrDevice s) {
    return func(a, GetReduceAxes(std::move(axis), a.ndim()), s);
  };
}

// A template converter for ops that accept |axis| and |keepdims|.
inline
std::function<mx::array(const mx::array& a,
                        IntOrVector axis,
                        std::optional<bool> keepdims,
                        mx::StreamOrDevice s)>
DimOpWrapper(mx::array(*func)(const mx::array&,
                              const std::vector<int>&,
                              bool,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                IntOrVector axis,
                std::optional<bool> keepdims,
                mx::StreamOrDevice s) {
    return func(a, GetReduceAxes(std::move(axis), a.ndim()),
                keepdims.value_or(false), s);
  };
}

// A template converter for |cum| ops.
inline
std::function<mx::array(const mx::array& a,
                        std::optional<int> axis,
                        std::optional<bool> reverse,
                        std::optional<bool> inclusive,
                        mx::StreamOrDevice s)>
CumOpWrapper(mx::array(*func)(const mx::array&,
                              int,
                              bool,
                              bool,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                std::optional<int> axis,
                std::optional<bool> reverse,
                std::optional<bool> inclusive,
                mx::StreamOrDevice s) {
    bool r = reverse.value_or(false);
    bool i = reverse.value_or(true);
    if (axis)
      return func(a, *axis, r, i, s);
    else
      return func(mx::reshape(a, {-1}, s), 0, r, i, s);
  };
}

#endif  // SRC_UTILS_H_
