#ifndef SRC_DEVICE_H_
#define SRC_DEVICE_H_

#include "src/bindings.h"

namespace ki {

template<>
struct Type<mx::Device::DeviceType> {
  static constexpr const char* name = "DeviceType";
  static napi_status ToNode(napi_env env,
                            mx::Device::DeviceType type,
                            napi_value* result);
  static std::optional<mx::Device::DeviceType> FromNode(napi_env env,
                                                        napi_value value);
};

template<>
struct Type<mx::Device> : public AllowPassByValue<mx::Device> {
  static constexpr const char* name = "Device";
  static mx::Device* Constructor(mx::Device::DeviceType type, int index);
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype);
  static std::optional<mx::Device> FromNode(napi_env env,
                                            napi_value value);
};

}  // namespace ki

#endif  // SRC_DEVICE_H_
