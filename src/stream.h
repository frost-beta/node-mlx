#ifndef SRC_STREAM_H_
#define SRC_STREAM_H_

#include "src/device.h"

namespace ki {

template<>
struct Type<mx::Stream> : public AllowPassByValue<mx::Stream> {
  static constexpr const char* name = "Stream";
  static mx::Stream* Constructor(int index, const mx::Device& device);
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype);
};

}  // namespace ki

#endif  // SRC_STREAM_H_
