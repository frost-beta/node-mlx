#include "src/bindings.h"

namespace {

napi_value Init(napi_env env, napi_value exports) {
  InitArray(env, exports);
  return exports;
}

}  // namespace

NAPI_MODULE(mlx, Init)
