#ifndef SRC_BINDINGS_H_
#define SRC_BINDINGS_H_

#include <mlx/mlx.h>
#include <kizunapi.h>

namespace mx = mlx::core;

void InitDevice(napi_env env, napi_value exports);
void InitStream(napi_env env, napi_value exports);
void InitArray(napi_env env, napi_value exports);
void InitMemory(napi_env env, napi_value exports);
void InitMetal(napi_env env, napi_value exports);
void InitOps(napi_env env, napi_value exports);
void InitIO(napi_env env, napi_value exports);
void InitTransforms(napi_env env, napi_value exports);
void InitRandom(napi_env env, napi_value exports);
void InitFFT(napi_env env, napi_value exports);
void InitLinalg(napi_env env, napi_value exports);
void InitConstants(napi_env env, napi_value exports);
void InitFast(napi_env env, napi_value exports);
void InitIndexing(napi_env env, napi_value exports);

#endif  // SRC_BINDINGS_H_
