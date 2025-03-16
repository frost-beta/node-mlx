#include "src/stream.h"
#include "src/utils.h"

namespace ki {

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
  DefineToString<mx::Stream>(env, prototype);
}

}  // namespace ki

namespace {

void Synchronize(std::optional<mx::Stream> s) {
  if (s)
    return mx::synchronize(*s);
  else
    return mx::synchronize();
}

}  // namespace

void InitStream(napi_env env, napi_value exports) {
  using to_stream_fun = mx::Stream (*)(mx::StreamOrDevice);
  to_stream_fun to_stream = &mx::to_stream;
  ki::Set(env, exports,
          "Stream", ki::Class<mx::Stream>(),
          "defaultStream", mx::default_stream,
          "setDefaultStream", mx::set_default_stream,
          "newStream", mx::new_stream,
          "toStream", to_stream,
          "synchronize", &Synchronize);
}
