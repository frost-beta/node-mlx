#include "src/bindings.h"

#include <limits>

void InitConstants(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "Inf", std::numeric_limits<double>::infinity(),
          "Infinity", std::numeric_limits<double>::infinity(),
          "inf", std::numeric_limits<double>::infinity(),
          "infty", std::numeric_limits<double>::infinity(),
          "NAN", NAN,
          "NaN", NAN,
          "nan", NAN,
          "NINF", -std::numeric_limits<double>::infinity(),
          "NZERO", -0.0,
          "PINF", std::numeric_limits<double>::infinity(),
          "PZERO", 0.0,
          "e", 2.71828182845904523536028747135266249775724709369995,
          "eulerGamma", 0.5772156649015328606065120900824024310421,
          "pi", 3.1415926535897932384626433,
          "newaxis", nullptr);
}
