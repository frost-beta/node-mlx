#include "src/array.h"
#include "src/stream.h"

namespace fast_ops {

mx::array Rope(const mx::array& x,
               int dims,
               bool traditional,
               std::optional<float> base,
               float scale,
               std::variant<int, mx::array> offset,
               const std::optional<mx::array>& freqs,
               mx::StreamOrDevice s) {
  if (std::holds_alternative<int>(offset)) {
    return mx::fast::rope(x, dims, traditional, base, scale,
                          std::get<int>(offset), freqs, s);
  } else {
    return mx::fast::rope(x, dims, traditional, base, scale,
                          std::get<mx::array>(offset), freqs, s);
  }
}

}  // namespace fast_ops

void InitFast(napi_env env, napi_value exports) {
  napi_value fast = ki::CreateObject(env);
  ki::Set(env, exports, "fast", fast);

  ki::Set(env, fast,
          "rmsNorm", &mx::fast::rms_norm,
          "layerNorm", &mx::fast::layer_norm,
          "rope", &fast_ops::Rope,
          "scaledDotProductAttention", &mx::fast::scaled_dot_product_attention);
}
