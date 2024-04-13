#include "src/bindings.h"

namespace {

napi_value Init(napi_env env, napi_value exports) {
  InitDevice(env, exports);
  InitStream(env, exports);
  InitArray(env, exports);
  InitOps(env, exports);
  return exports;
}

}  // namespace

NAPI_MODULE(mlx, Init)
