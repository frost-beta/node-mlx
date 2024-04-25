#include "src/complex.h"

namespace ki {

// static
napi_status Type<std::complex<float>>::ToNode(napi_env env,
                                              std::complex<float> value,
                                              napi_value* result) {
  napi_status s = napi_create_object(env, result);
  if (s != napi_ok)
    return s;
  Set(env, *result, "re", value.real(), "im", value.imag());
  return napi_ok;
}

// static
std::optional<std::complex<float>> Type<std::complex<float>>::FromNode(
    napi_env env,
    napi_value value) {
  float real, imag;
  if (!Get(env, value, "re", &real, "im", &imag))
    return std::nullopt;
  return std::complex<float>(real, imag);
}

}  // namespace ki

bool IsComplexNumber(napi_env env, napi_value value) {
  bool has = false;
  napi_has_property(env, value, ki::ToNodeValue(env, "re"), &has);
  return has;
}
