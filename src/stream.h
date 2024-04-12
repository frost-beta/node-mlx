#ifndef SRC_STREAM_H_
#define SRC_STREAM_H_

#include "src/device.h"

namespace ki {

template<>
struct Type<mx::Stream> {
  static constexpr const char* name = "Stream";

  static mx::Stream* Constructor(int index, const mx::Device& device);
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype);

  static napi_status ToNode(napi_env env,
                            mx::Stream stream,
                            napi_value* result);
  static std::optional<mx::Stream> FromNode(napi_env env,
                                            napi_value value);
};

template<>
struct Type<mx::StreamOrDevice> {
  static constexpr const char* name = "StreamOrDevice";
  static std::optional<mx::StreamOrDevice> FromNode(napi_env env,
                                                    napi_value value);
};

}  // namespace ki

#endif  // SRC_STREAM_H_
