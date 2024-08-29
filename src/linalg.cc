#include "src/array.h"
#include "src/stream.h"

namespace linalg {

mx::array Norm(const mx::array& a,
               std::optional<std::variant<float, std::string>> ord,
               std::optional<std::variant<int, std::vector<int>>> optional_axis,
               std::optional<bool> optional_keepdims,
               mx::StreamOrDevice s) {
  std::optional<std::vector<int>> axis;
  if (optional_axis)
    axis = PutIntoVector(std::move(*optional_axis));
  bool keepdims = optional_keepdims.value_or(false);

  if (!ord) {
    return mx::linalg::norm(a, std::move(axis), keepdims, s);
  } else if (auto str = std::get_if<std::string>(&ord.value()); str) {
    return mx::linalg::norm(a, std::move(*str), std::move(axis), keepdims, s);
  } else {
    return mx::linalg::norm(a, std::get<float>(*ord), std::move(axis), keepdims,
                            s);
  }
}

}  // namespace linalg

void InitLinalg(napi_env env, napi_value exports) {
  napi_value linalg = ki::CreateObject(env);
  ki::Set(env, exports, "linalg", linalg);

  ki::Set(env, linalg,
          "norm", &linalg::Norm,
          "qr", &mx::linalg::qr,
          "svd", &mx::linalg::svd,
          "inv", &mx::linalg::inv,
          "triInv", &mx::linalg::tri_inv,
          "cholesky", &mx::linalg::cholesky,
          "choleskyInv", &mx::linalg::cholesky_inv);
}
