#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_

#include <numeric>

#include <mlx/mlx.h>
#include <kizunapi.h>

namespace mx = mlx::core;

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
using IntOrVec = std::variant<std::monostate, int, std::vector<int>>;
inline std::vector<int> GetReduceAxes(IntOrVec value, int dims) {
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

// A template converter for ops that accept |axis|.
inline
std::function<mx::array(const mx::array& a,
                        IntOrVec axis,
                        mx::StreamOrDevice s)>
DimOpWrapper(mx::array(*func)(const mx::array&,
                              const std::vector<int>&,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                IntOrVec axis,
                mx::StreamOrDevice s) {
    return func(a, GetReduceAxes(std::move(axis), a.ndim()), s);
  };
}

// A template converter for ops that accept |axis| and |keepdims|.
inline
std::function<mx::array(const mx::array& a,
                        IntOrVec axis,
                        std::optional<bool> keepdims,
                        mx::StreamOrDevice s)>
DimOpWrapper(mx::array(*func)(const mx::array&,
                              const std::vector<int>&,
                              bool,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                IntOrVec axis,
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

#endif  // SRC_UTIL_H_
