#include "src/array.h"
#include "src/utils.h"

std::vector<int> GetReduceAxes(IntOrVector value, int dims) {
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

mx::array ToArray(ScalarOrArray value, std::optional<mx::Dtype> dtype) {
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

size_t ReadArgsToArrays(ki::Arguments* args, std::vector<mx::array>* results) {
  for (size_t i = 0; i < args->Length(); ++i) {
    std::optional<mx::array> a = args->GetNext<mx::array>();
    if (!a) {
      args->ThrowError("array");
      return i;
    }
    results->push_back(std::move(*a));
  }
  return args->Length();
}
