#ifndef SRC_COMPLEX_H_
#define SRC_COMPLEX_H_

#include <kizunapi.h>

#include <complex>

namespace ki {

template<>
struct Type<std::complex<float>> {
  static constexpr const char* name = "complex64";
  static napi_status ToNode(napi_env env,
                            std::complex<float> value,
                            napi_value* result);
  static std::optional<std::complex<float>> FromNode(napi_env env,
                                                 napi_value value);
};

}  // namespace ki

bool IsComplexNumber(napi_env env, napi_value value);

#endif  // SRC_COMPLEX_H_
