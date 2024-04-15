#include "src/device.h"
#include "src/utils.h"

namespace ki {

// static
napi_status Type<mx::Device::DeviceType>::ToNode(
    napi_env env,
    mx::Device::DeviceType type,
    napi_value* result) {
  return ConvertToNode(env, static_cast<int>(type), result);
}

// static
std::optional<mx::Device::DeviceType> Type<mx::Device::DeviceType>::FromNode(
    napi_env env,
    napi_value value) {
  std::optional<int> type = FromNodeTo<int>(env, value);
  if (!type)
    return std::nullopt;
  if (*type == static_cast<int>(mx::Device::DeviceType::cpu))
    return mx::Device::DeviceType::cpu;
  if (*type == static_cast<int>(mx::Device::DeviceType::gpu))
    return mx::Device::DeviceType::gpu;
  return std::nullopt;
}

// static
mx::Device* Type<mx::Device>::Constructor(mx::Device::DeviceType type,
                                          int index) {
  return new mx::Device(type, index);
}

// static
void Type<mx::Device>::Define(napi_env env,
                              napi_value constructor,
                              napi_value prototype) {
  DefineProperties(env, prototype,
                   Property("type", Getter(&mx::Device::type)));
}

// static
std::optional<mx::Device> Type<mx::Device>::FromNode(napi_env env,
                                                     napi_value value) {
  // Try creating a Device when value is a DeviceType.
  if (auto t = FromNodeTo<mx::Device::DeviceType>(env, value); t)
    return mx::Device(*t);
  // Otherwise try converting from Device.
  return AllowPassByValue<mx::Device>::FromNode(env, value);
}

}  // namespace ki

void InitDevice(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "cpu", mx::Device::DeviceType::cpu,
          "gpu", mx::Device::DeviceType::gpu,
          "Device", ki::Class<mx::Device>(),
          "defaultDevice", mx::default_device,
          "setDefaultDevice", mx::set_default_device);
}
