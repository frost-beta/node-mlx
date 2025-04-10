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

mx::array ScaledDotProductAttention(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    const float scale,
    const std::variant<std::monostate, std::string, mx::array>& mask,
    mx::StreamOrDevice s) {
  bool has_mask = !std::holds_alternative<std::monostate>(mask);
  bool has_str_mask =
      has_mask && std::holds_alternative<std::string>(mask);
  bool has_arr_mask = has_mask && std::holds_alternative<mx::array>(mask);

  if (has_mask) {
    if (has_str_mask) {
      auto mask_str = std::get<std::string>(mask);
      if (mask_str != "causal") {
        std::ostringstream msg;
        msg << "[scaled_dot_product_attention] invalid mask option '"
            << mask_str << "'. Must be 'causal', or an array.";
        throw std::invalid_argument(msg.str());
      }
      return mx::fast::scaled_dot_product_attention(
          queries, keys, values, scale, mask_str, {}, s);
    } else {
      auto mask_arr = std::get<mx::array>(mask);
      return mx::fast::scaled_dot_product_attention(
          queries, keys, values, scale, "", {mask_arr}, s);
    }

  } else {
    return mx::fast::scaled_dot_product_attention(
        queries, keys, values, scale, "", {}, s);
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
          "scaledDotProductAttention", &fast_ops::ScaledDotProductAttention);
}
