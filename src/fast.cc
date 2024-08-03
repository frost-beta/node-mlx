#include "src/array.h"
#include "src/stream.h"

namespace fast_ops {

mx::array AffineQuantize(const mx::array& w,
                         const mx::array& scales,
                         const mx::array& biases,
                         std::optional<int> group_size,
                         std::optional<int> bits,
                         mx::StreamOrDevice s) {
  return mx::fast::affine_quantize(w, scales, biases, group_size.value_or(64),
                                   bits.value_or(4));
}

}  // namespace fast_ops

void InitFast(napi_env env, napi_value exports) {
  napi_value fast = ki::CreateObject(env);
  ki::Set(env, exports, "fast", fast);

  ki::Set(env, fast,
          "rmsNorm", &mx::fast::rms_norm,
          "layerNorm", &mx::fast::layer_norm,
          "rope", &mx::fast::rope,
          "scaledDotProductAttention", &mx::fast::scaled_dot_product_attention,
          "affineQuantize", &fast_ops::AffineQuantize);
}
