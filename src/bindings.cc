#include "src/bindings.h"

namespace {

napi_value Init(napi_env env, napi_value exports) {
  InitDevice(env, exports);
  InitStream(env, exports);
  InitArray(env, exports);
  InitMemory(env, exports);
  InitMetal(env, exports);
  InitOps(env, exports);
  InitIO(env, exports);
  InitTransforms(env, exports);
  InitRandom(env, exports);
  InitFFT(env, exports);
  InitLinalg(env, exports);
  InitConstants(env, exports);
  InitFast(env, exports);
  InitIndexing(env, exports);
  return exports;
}

}  // namespace

NAPI_MODULE(mlx, Init)
