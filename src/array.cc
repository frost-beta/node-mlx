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
  // Since Dtype is represented as a class, we have to store it as a pointer in
  // js, so converting it to js usually would involve a heap allocation. To
  // avoid that let's just find the global const.
  static inline napi_status ToNode(napi_env env,
                                   const mx::Dtype& value,
                                   napi_value* result) {
    if (value == mx::bool_)
      return ConvertToNode(env, &mx::bool_, result);
    if (value == mx::uint8)
      return ConvertToNode(env, &mx::uint8, result);
    if (value == mx::uint16)
      return ConvertToNode(env, &mx::uint16, result);
    if (value == mx::uint32)
      return ConvertToNode(env, &mx::uint32, result);
    if (value == mx::uint64)
      return ConvertToNode(env, &mx::uint64, result);
    if (value == mx::int8)
      return ConvertToNode(env, &mx::int8, result);
    if (value == mx::int16)
      return ConvertToNode(env, &mx::int16, result);
    if (value == mx::int32)
      return ConvertToNode(env, &mx::int32, result);
    if (value == mx::int64)
      return ConvertToNode(env, &mx::int64, result);
    if (value == mx::float16)
      return ConvertToNode(env, &mx::float16, result);
    if (value == mx::float32)
      return ConvertToNode(env, &mx::float32, result);
    if (value == mx::bfloat16)
      return ConvertToNode(env, &mx::bfloat16, result);
    if (value == mx::complex64)
      return ConvertToNode(env, &mx::complex64, result);
    return napi_generic_failure;
  }
  // Dtype is stored as pointer in js, convert it to value in C++ by copy.
  static inline std::optional<mx::Dtype> FromNode(napi_env env,
                                                  napi_value value) {
    std::optional<mx::Dtype*> ptr = ki::FromNode<mx::Dtype*>(env, value);
    if (!ptr)
      return std::nullopt;
    return *ptr.value();
  }
};

// Allow passing Dtype to js directly, no memory management involved as they are
// static globals.
template<>
struct TypeBridge<mx::Dtype> {
  static inline mx::Dtype* Wrap(mx::Dtype* ptr) {
    return ptr;
  }
  static inline void Finalize(mx::Dtype* ptr) {
  }
};

template<>
struct Type<mx::array> {
  static constexpr const char* name = "array";
  static mx::array* Constructor(napi_env env,
                                napi_value value,
                                std::optional<mx::Dtype> dtype) {
    napi_valuetype type;
    if (napi_typeof(env, value, &type) != napi_ok)
      return nullptr;
    switch (type) {
      case napi_boolean:
        return new mx::array(ki::FromNode<bool>(env, value).value(),
                             dtype.value_or(mx::bool_));
      case napi_number:
        return new mx::array(ki::FromNode<float>(env, value).value(),
                             dtype.value_or(mx::float32));
      default:
        return nullptr;
    }
  }
  static inline void Destructor(mx::array* ptr) {
    delete ptr;
  }
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype) {
    // Disambiguate the 2 overloads of shape().
    using shape_fun = const std::vector<int>& (mx::array::*)() const;
    shape_fun shape = &mx::array::shape;
    // Define array's properties.
    DefineProperties(env, prototype,
                     Property("size", Getter(&mx::array::size)),
                     Property("ndim", Getter(&mx::array::ndim)),
                     Property("itemsize", Getter(&mx::array::itemsize)),
                     Property("nbytes", Getter(&mx::array::nbytes)),
                     Property("shape", Getter(shape)),
                     Property("dtype", Getter(&mx::array::dtype)));
    // Define array's methods.
    Set(env, prototype,
        "item", MemberFunction(&Item));
  }
  static napi_value Item(mx::array* a, napi_env env) {
    a->eval();
    switch (a->dtype()) {
      case mx::bool_:
        return ToNode(env, a->item<bool>());
      case mx::uint8:
        return ToNode(env, a->item<uint8_t>());
      case mx::uint16:
        return ToNode(env, a->item<uint16_t>());
      case mx::uint32:
        return ToNode(env, a->item<uint32_t>());
      case mx::uint64:
        return ToNode(env, a->item<uint64_t>());
      case mx::int8:
        return ToNode(env, a->item<int8_t>());
      case mx::int16:
        return ToNode(env, a->item<int16_t>());
      case mx::int32:
        return ToNode(env, a->item<int32_t>());
      case mx::int64:
        return ToNode(env, a->item<int64_t>());
      case mx::float16:
        return ToNode(env, static_cast<float>(a->item<mx::float16_t>()));
      case mx::float32:
        return ToNode(env, a->item<float>());
      case mx::bfloat16:
        return ToNode(env, static_cast<float>(a->item<mx::bfloat16_t>()));
      case mx::complex64:
        // FIXME(zcbenz): Represent complex number in JS.
        return Undefined(env);
    }
  }
  // array is stored as pointer in js, convert it to value in C++ by copy.
  static inline std::optional<mx::array> FromNode(napi_env env,
                                                  napi_value value) {
    std::optional<mx::array*> ptr = ki::FromNode<mx::array*>(env, value);
    if (!ptr)
      return std::nullopt;
    return *ptr.value();
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

  ki::Set(env, exports,
          "array", ki::Class<mx::array>());
}
