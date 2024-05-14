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
std::string ToString(const T* value) {
  std::ostringstream ss;
  ss << *value;
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

#endif  // SRC_UTILS_H_
