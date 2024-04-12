#include "src/stream.h"
#include "src/util.h"

namespace ki {

template<>
struct TypeBridge<mx::Stream> {
  static inline void Finalize(mx::Stream* ptr) {
    delete ptr;
  }
};

// static
mx::Stream* Type<mx::Stream>::Constructor(int index, const mx::Device& device) {
  return new mx::Stream(index, device);
}

// static
void Type<mx::Stream>::Define(napi_env env,
                              napi_value constructor,
                              napi_value prototype) {
  DefineProperties(env, prototype,
                   Property("device", Getter(&mx::Stream::device)));
}

// static
napi_status Type<mx::Stream>::ToNode(napi_env env,
                                     mx::Stream stream,
                                     napi_value* result) {
  return ManagePointerInJSWrapper(
      env, new mx::Stream(std::move(stream)), result);
}

// static
std::optional<mx::Stream> Type<mx::Stream>::FromNode(napi_env env,
                                                     napi_value value) {
  return NodeObjToCppValue<mx::Stream>(env, value);
}

// static
std::optional<mx::StreamOrDevice> Type<mx::StreamOrDevice>::FromNode(
    napi_env env,
    napi_value value) {
  std::optional<mx::Stream> stream = Type<mx::Stream>::FromNode(env, value);
  if (stream)
    return *stream;
  std::optional<mx::Device> device = Type<mx::Device>::FromNode(env, value);
  if (device)
    return *device;
  return std::nullopt;
}

}  // namespace ki

void InitStream(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "Stream", ki::Class<mx::Stream>(),
          "defaultStream", mx::default_stream,
          "setDefaultStream", mx::set_default_stream,
          "newStream", mx::new_stream);
}
