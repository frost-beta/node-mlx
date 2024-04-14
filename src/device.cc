#include "src/device.h"
#include "src/utils.h"

namespace ki {

template<>
struct TypeBridge<mx::Device> {
  static inline void Finalize(mx::Device* ptr) {
    delete ptr;
  }
};

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
  std::optional<int> type = ki::FromNode<int>(env, value);
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
napi_status Type<mx::Device>::ToNode(napi_env env,
                                     mx::Device device,
                                     napi_value* result) {
  return ManagePointerInJSWrapper(
      env, new mx::Device(std::move(device)), result);
}

// static
std::optional<mx::Device> Type<mx::Device>::FromNode(napi_env env,
                                                     napi_value value) {
  // Try creating a Device when value is a DeviceType.
  auto type = ki::FromNode<mx::Device::DeviceType>(env, value);
  if (type)
    return mx::Device(*type);
  // Otherwise try converting from Device.
  return NodeObjToCppValue<mx::Device>(env, value);
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
