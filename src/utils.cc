#include "src/array.h"
#include "src/utils.h"

std::vector<int> PutIntoVector(std::variant<int, std::vector<int>> shape) {
  if (auto i = std::get_if<int>(&shape); i)
    return {*i};
  return std::move(std::get<std::vector<int>>(shape));
}

std::vector<int> GetReduceAxes(OptionalAxes value, int dims) {
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

napi_value AwaitFunction(
    napi_env env,
    std::function<napi_value()> func,
    std::function<napi_value(napi_env, napi_value)> cpp_then,
    std::function<void(napi_env)> cpp_finally) {
  napi_value result = func();
  // Return immediately if the result is not promise.
  bool is_promise = false;
  napi_is_promise(env, result, &is_promise);
  if (!is_promise) {
    cpp_then(env, result);
    cpp_finally(env);
    return result;
  }
  // Pass the callbacks to promise.
  napi_value then, finally;
  if (!ki::Get(env, result, "then", &then, "finally", &finally)) {
    ki::ThrowError(env, "No then and finally method in Promise.");
    return nullptr;
  }
  napi_value js_then = ki::ToNodeValue(env, cpp_then);
  napi_make_callback(env, nullptr, result, then, 1, &js_then, &result);
  napi_value js_finally = ki::ToNodeValue(env, cpp_finally);
  napi_make_callback(env, nullptr, result, finally, 1, &js_finally, &result);
  return result;
}
