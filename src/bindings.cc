#include <kizunapi.h>

namespace {

napi_value Init(napi_env env, napi_value exports) {
  return exports;
}

}  // namespace

NAPI_MODULE(mlx, Init)
