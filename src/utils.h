#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <numeric>

#include <mlx/mlx.h>
#include <kizunapi.h>

namespace mx = mlx::core;

using IntOrVector = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::variant<bool, float, mx::array>;

// Convert a int or vector into vector.
inline std::vector<int> ToIntVector(std::variant<int, std::vector<int>> shape) {
  if (auto i = std::get_if<int>(&shape); i)
    return {*i};
  return std::move(std::get<std::vector<int>>(shape));
}

// Get axis arg from js value.
std::vector<int> GetReduceAxes(IntOrVector value, int dims);

// Convert a ScalarOrArray arg to array.
mx::array ToArray(ScalarOrArray value,
                  std::optional<mx::Dtype> dtype = std::nullopt);

// Read args into a vector of arrays.
size_t ReadArgsToArrays(ki::Arguments* args, std::vector<mx::array>* results);

#endif  // SRC_UTILS_H_
