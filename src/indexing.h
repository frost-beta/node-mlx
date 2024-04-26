#ifndef SRC_INDEXING_H_
#define SRC_INDEXING_H_

#include "src/bindings.h"

struct Slice {
  std::optional<int> start;
  std::optional<int> stop;
  std::optional<int> step;
};

struct Ellipsis {};

using ArrayIndex = std::variant<std::monostate,  // null/newaxis
                                Ellipsis,  // ...
                                Slice,  // start:stop:step
                                mx::array*,
                                int>;
using ArrayIndices = std::vector<ArrayIndex>;

mx::array Index(mx::array* a, ki::Arguments* args);

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
