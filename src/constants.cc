#include "src/bindings.h"

#include <limits>

void InitConstants(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "e", 2.71828182845904523536028747135266249775724709369995,
          "eulerGamma", 0.5772156649015328606065120900824024310421,
          "inf", std::numeric_limits<double>::infinity(),
          "nan", NAN,
          "pi", 3.1415926535897932384626433,
          "newaxis", nullptr);
}
