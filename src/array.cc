#include "mlx/mlx.h"
#include "src/bindings.h"

namespace mx = mlx::core;

namespace ki {

template<>
struct Type<mx::Dtype> {
  static constexpr const char* name = "Dtype";
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype) {
    DefineProperties(env, prototype,
                     Property("size", Getter(&mx::Dtype::size)));
  }
};

template<>
struct TypeBridge<mx::Dtype> {
  static inline mx::Dtype* Wrap(mx::Dtype* ptr) {
    return ptr;
  }
  static inline void Finalize(mx::Dtype* ptr) {
  }
};

}  // namespace ki

void InitArray(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "Dtype", ki::Class<mx::Dtype>(),
          "bool_", &mx::bool_,
          "uint8", &mx::uint8,
          "uint16", &mx::uint16,
          "uint32", &mx::uint32,
          "uint64", &mx::uint64,
          "int8", &mx::int8,
          "int16", &mx::int16,
          "int32", &mx::int32,
          "int64", &mx::int64,
          "float16", &mx::float16,
          "float32", &mx::float32,
          "bfloat16", &mx::bfloat16,
          "complex64", &mx::complex64);
}
