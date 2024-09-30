#ifndef SRC_ARRAY_H_
#define SRC_ARRAY_H_

#include "src/utils.h"

namespace ki {

template<>
struct Type<mx::float16_t> {
  static constexpr const char* name = "mx.float16";
  static inline napi_status ToNode(napi_env env,
                                   mx::float16_t value,
                                   napi_value* result) {
    return ConvertToNode(env, static_cast<float>(value), result);
  }
};

template<>
struct Type<mx::bfloat16_t> {
  static constexpr const char* name = "mx.bfloat16";
  static inline napi_status ToNode(napi_env env,
                                   mx::bfloat16_t value,
                                   napi_value* result) {
    return ConvertToNode(env, static_cast<float>(value), result);
  }
};

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
struct Type<mx::Dtype::Category> {
  static constexpr const char* name = "DtypeCategory";
  static napi_status ToNode(napi_env env,
                            mx::Dtype::Category value,
                            napi_value* result);
  static std::optional<mx::Dtype::Category> FromNode(napi_env env,
                                                     napi_value value);
};

template<>
struct Type<mx::array> : public AllowPassByValue<mx::array> {
  static constexpr const char* name = "mx.array";
  static constexpr bool allow_function_call = true;
  static mx::array* Constructor(napi_env env,
                                napi_value value,
                                std::optional<mx::Dtype> dtype);
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype);
  // The default FromNode method only accepts array instance, with the custom
  // FromNode converter we can pass scalars to ops directly, making calls like
  // mx.equal(1, 2) possible.
  static std::optional<mx::array> FromNode(napi_env env,
                                           napi_value value);
};

}  // namespace ki

#endif  // SRC_ARRAY_H_
