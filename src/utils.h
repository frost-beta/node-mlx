#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <numeric>

#include <mlx/mlx.h>
#include <kizunapi.h>

namespace mx = mlx::core;

using OptionalAxes = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::variant<bool, float, mx::array>;

// Read args into a vector of types.
template<typename T>
bool ReadArgs(ki::Arguments* args, std::vector<T>* results) {
  while (args->RemainingsLength() > 0) {
    std::optional<T> a = args->GetNext<T>();
    if (!a) {
      args->ThrowError(ki::Type<T>::name);
      return false;
    }
    results->push_back(std::move(*a));
  }
  return true;
}

// Convert the type to string.
template<typename T>
std::string ToString(napi_value value, napi_env env) {
  std::optional<T*> self = ki::FromNodeTo<T*>(env, value);
  if (!self)
    return std::string("The object has been destroyed.");
  std::ostringstream ss;
  ss << *self.value();
  return ss.str();
}

// Define the toString method for type's prototype.
template<typename T>
void DefineToString(napi_env env, napi_value prototype) {
  auto symbol = ki::SymbolFor("nodejs.util.inspect.custom");
  ki::Set(env, prototype,
          "toString", ki::MemberFunction(&ToString<T>),
          symbol, ki::MemberFunction(&ToString<T>));
}

// If input is one int, put it into a vector, otherwise just return the vector.
std::vector<int> PutIntoVector(std::variant<int, std::vector<int>> shape);

// Get axis arg from js value.
std::vector<int> GetReduceAxes(OptionalAxes value, int dims);

// Convert a ScalarOrArray arg to array.
mx::array ToArray(ScalarOrArray value,
                  std::optional<mx::Dtype> dtype = std::nullopt);

// Execute the function and wait it to finish.
napi_value AwaitFunction(
    napi_env env,
    std::function<napi_value()> func,
    std::function<napi_value(napi_env, napi_value)> cpp_then,
    std::function<void(napi_env)> cpp_finally);

#endif  // SRC_UTILS_H_
