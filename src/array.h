#ifndef SRC_ARRAY_H_
#define SRC_ARRAY_H_

#include "src/utils.h"

namespace ki {

template<>
struct Type<mx::Dtype> {
  static constexpr const char* name = "Dtype";
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype);
  static napi_status ToNode(napi_env env,
                            const mx::Dtype& value,
                            napi_value* result);
  static std::optional<mx::Dtype> FromNode(napi_env env,
                                           napi_value value);
};

template<>
struct Type<mx::array> {
  static constexpr const char* name = "array";
  static mx::array* Constructor(napi_env env,
                                napi_value value,
                                std::optional<mx::Dtype> dtype);
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype);
  static napi_status ToNode(napi_env env,
                            mx::array a,
                            napi_value* result);
  static std::optional<mx::array> FromNode(napi_env env,
                                           napi_value value);

 private:
  static napi_value Item(mx::array* a, napi_env env);
  static mx::array Var(mx::array* a,
                       IntOrVector axis,
                       std::optional<bool> keepdims,
                       std::optional<int> ddof,
                       mx::StreamOrDevice s);
  static std::vector<mx::array> Split(
      mx::array* a,
      std::variant<int, std::vector<int>> indices,
      std::optional<int> axis,
      mx::StreamOrDevice s);
  static mx::array ArgMin(mx::array* a,
                          std::optional<int> axis,
                          std::optional<bool> keepdims,
                          mx::StreamOrDevice s);
  static mx::array ArgMax(mx::array* a,
                          std::optional<int> axis,
                          std::optional<bool> keepdims,
                          mx::StreamOrDevice s);
};

}  // namespace ki

#endif  // SRC_ARRAY_H_
