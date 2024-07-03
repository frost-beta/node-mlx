#ifndef SRC_INDEXING_H_
#define SRC_INDEXING_H_

#include "src/bindings.h"

struct Slice {
  std::optional<int> start;
  std::optional<int> stop;
  std::optional<int> step;
};

struct Ellipsis {};

// TODO(zcbenz): Accept int[] as index.
// https://github.com/ml-explore/mlx/pull/1150
using ArrayIndex = std::variant<std::monostate,  // null/newaxis
                                Ellipsis,  // ...
                                Slice,  // start:stop:step
                                int,
                                mx::array*>;
using ArrayIndices = std::vector<ArrayIndex>;

class ArrayAt {
 public:
  ArrayAt(mx::array x, std::variant<ArrayIndex, ArrayIndices> indices);

  mx::array Add(ScalarOrArray value);
  mx::array Subtract(ScalarOrArray value);
  mx::array Multiply(ScalarOrArray value);
  mx::array Divide(ScalarOrArray value);
  mx::array Maximum(ScalarOrArray value);
  mx::array Minimum(ScalarOrArray value);

 private:
  mx::array x_;
  std::variant<ArrayIndex, ArrayIndices> indices_;
};

mx::array Index(const mx::array* a, ki::Arguments* args);
void IndexPut(mx::array* a,
              std::variant<ArrayIndex, ArrayIndices> obj,
              ScalarOrArray vals);

namespace ki {

template<>
struct Type<Slice> {
  static constexpr const char* name = "Slice";
  static napi_status ToNode(napi_env env,
                            const Slice& value,
                            napi_value* result);
  static std::optional<Slice> FromNode(napi_env env,
                                       napi_value value);
};

template<>
struct Type<Ellipsis> {
  static constexpr const char* name = "Ellipsis";
  static napi_status ToNode(napi_env env,
                            const Ellipsis& value,
                            napi_value* result);
  static std::optional<Ellipsis> FromNode(napi_env env,
                                          napi_value value);
};

}  // namespace ki

#endif  // SRC_INDEXING_H_
