#include "src/array.h"
#include "src/stream.h"

void InitFast(napi_env env, napi_value exports) {
  napi_value fast = ki::CreateObject(env);
  ki::Set(env, exports, "fast", fast);

  ki::Set(env, fast,
          "rmsNorm", &mx::fast::rms_norm,
          "layerNorm", &mx::fast::layer_norm,
          "rope", &mx::fast::rope,
          "scaledDotProductAttention", &mx::fast::scaled_dot_product_attention);
}
