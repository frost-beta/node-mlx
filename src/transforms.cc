#include "src/array.h"

namespace ops {

void Eval(ki::Arguments* args) {
  std::vector<mx::array> arrays;
  for (size_t i = 0; i < args->Length(); ++i) {
    std::optional<mx::array> a = args->GetNext<mx::array>();
    if (!a) {
      args->ThrowError("array");
      return;
    }
    arrays.push_back(std::move(*a));
  }
  mx::eval(std::move(arrays));
}

}  // namespace ops

void InitTransforms(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "eval", &ops::Eval);
}
