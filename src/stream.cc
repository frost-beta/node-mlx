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
}

}  // namespace ki

void InitStream(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "Stream", ki::Class<mx::Stream>(),
          "defaultStream", mx::default_stream,
          "setDefaultStream", mx::set_default_stream,
          "newStream", mx::new_stream,
          "toStream", mx::to_stream);
}
