#include "src/array.h"
#include "src/ops.h"

namespace ki {

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

// Define finalize for array, which is called for arrays created from js, and
// arrays passed from C++ to js.
template<>
struct TypeBridge<mx::array> {
  static inline void Finalize(mx::array* ptr) {
    delete ptr;
  }
};

// static
void Type<mx::Dtype>::Define(napi_env env,
                             napi_value constructor,
                             napi_value prototype) {
  DefineProperties(env, prototype,
                   Property("size", Getter(&mx::Dtype::size)));
}

// static
napi_status Type<mx::Dtype>::ToNode(napi_env env,
                                    const mx::Dtype& value,
                                    napi_value* result) {
  // Since Dtype is represented as a class, we have to store it as a pointer
  // in js, so converting it to js usually would involve a heap allocation. To
  // avoid that let's just find the global const.
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

// static
std::optional<mx::Dtype> Type<mx::Dtype>::FromNode(napi_env env,
                                                   napi_value value) {
  return NodeObjToCppValue<mx::Dtype>(env, value);
}

// static
napi_status Type<mx::Dtype::Category>::ToNode(
    napi_env env, mx::Dtype::Category type, napi_value* result) {
  return ConvertToNode(env, static_cast<int>(type), result);
}

// static
std::optional<mx::Dtype::Category> Type<mx::Dtype::Category>::FromNode(
    napi_env env, napi_value value) {
  std::optional<int> type = ki::FromNode<int>(env, value);
  if (!type)
    return std::nullopt;
  if (*type == static_cast<int>(mx::Dtype::Category::complexfloating))
    return mx::Dtype::Category::complexfloating;
  if (*type == static_cast<int>(mx::Dtype::Category::floating))
    return mx::Dtype::Category::floating;
  if (*type == static_cast<int>(mx::Dtype::Category::inexact))
    return mx::Dtype::Category::inexact;
  if (*type == static_cast<int>(mx::Dtype::Category::signedinteger))
    return mx::Dtype::Category::signedinteger;
  if (*type == static_cast<int>(mx::Dtype::Category::unsignedinteger))
    return mx::Dtype::Category::unsignedinteger;
  if (*type == static_cast<int>(mx::Dtype::Category::integer))
    return mx::Dtype::Category::integer;
  if (*type == static_cast<int>(mx::Dtype::Category::number))
    return mx::Dtype::Category::number;
  if (*type == static_cast<int>(mx::Dtype::Category::generic))
    return mx::Dtype::Category::generic;
  return std::nullopt;
}

// static
mx::array* Type<mx::array>::Constructor(napi_env env,
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

// static
void Type<mx::array>::Define(napi_env env,
                             napi_value constructor,
                             napi_value prototype) {
  // Disambiguate the 2 overloads of shape().
  using shape_fun = const std::vector<int>& (mx::array::*)() const;
  shape_fun shape = &mx::array::shape;
  // Disambiguate the 3 overloads of transpose().
  using t_fun = mx::array (*)(const mx::array&, mx::StreamOrDevice);
  t_fun t = &mx::transpose;
  // Define array's properties.
  DefineProperties(env, prototype,
                   Property("size", Getter(&mx::array::size)),
                   Property("ndim", Getter(&mx::array::ndim)),
                   Property("itemsize", Getter(&mx::array::itemsize)),
                   Property("nbytes", Getter(&mx::array::nbytes)),
                   Property("shape", Getter(shape)),
                   Property("dtype", Getter(&mx::array::dtype)),
                   Property("T", Getter(t)));
  // Define array's methods.
  Set(env, prototype,
      "item", MemberFunction(&Item),
      "astype", MemberFunction(&mx::astype),
      "flatten", MemberFunction(&ops::Flatten),
      "reshape", MemberFunction(&mx::reshape),
      "squeeze", MemberFunction(&ops::Squeeze),
      "abs", MemberFunction(&mx::abs),
      "square", MemberFunction(&mx::square),
      "sqrt", MemberFunction(&mx::sqrt),
      "rsqrt", MemberFunction(&mx::rsqrt),
      "reciprocal", MemberFunction(&mx::reciprocal),
      "exp", MemberFunction(&mx::exp),
      "log", MemberFunction(&mx::log),
      "log2", MemberFunction(&mx::log2),
      "log10", MemberFunction(&mx::log10),
      "sin", MemberFunction(&mx::sin),
      "cos", MemberFunction(&mx::cos),
      "log1p", MemberFunction(&mx::log1p),
      "all", MemberFunction(DimOpWrapper(&mx::all)),
      "any", MemberFunction(DimOpWrapper(&mx::any)),
      "moveaxis", MemberFunction(&mx::moveaxis),
      "transpose", MemberFunction(&ops::Transpose),
      "sum", MemberFunction(DimOpWrapper(&mx::sum)),
      "prod", MemberFunction(DimOpWrapper(&mx::prod)),
      "min", MemberFunction(DimOpWrapper(&mx::min)),
      "max", MemberFunction(DimOpWrapper(&mx::max)),
      "logsumexp", MemberFunction(DimOpWrapper(&mx::logsumexp)),
      "mean", MemberFunction(DimOpWrapper(&mx::mean)),
      "var", MemberFunction(&ops::Var),
      "split", MemberFunction(&ops::Split),
      "argmin", MemberFunction(&ops::ArgMin),
      "argmax", MemberFunction(&ops::ArgMax),
      "cumsum", MemberFunction(CumOpWrapper(&mx::cumsum)),
      "cumprod", MemberFunction(CumOpWrapper(&mx::cumprod)),
      "cummax", MemberFunction(CumOpWrapper(&mx::cummax)),
      "cummin", MemberFunction(CumOpWrapper(&mx::cummin)),
      "round", MemberFunction(&ops::Round),
      "diagonal", MemberFunction(&ops::Diagonal),
      "diag", MemberFunction(&ops::Diag));
}

// static
napi_status Type<mx::array>::ToNode(napi_env env,
                                    mx::array a,
                                    napi_value* result) {
  return ManagePointerInJSWrapper(env, new mx::array(std::move(a)), result);
}

// static
std::optional<mx::array> Type<mx::array>::FromNode(napi_env env,
                                                   napi_value value) {
  if (auto p = ki::FromNode<mx::array*>(env, value); p)
    return *p.value();
  if (auto b = ki::FromNode<bool>(env, value); b)
    return mx::array(*b, mx::bool_);
  if (auto f = ki::FromNode<float>(env, value); f)
    return mx::array(*f, mx::float32);
  return std::nullopt;
}

// static
napi_value Type<mx::array>::Item(mx::array* a, napi_env env) {
  a->eval();
  switch (a->dtype()) {
    case mx::bool_:
      return ki::ToNode(env, a->item<bool>());
    case mx::uint8:
      return ki::ToNode(env, a->item<uint8_t>());
    case mx::uint16:
      return ki::ToNode(env, a->item<uint16_t>());
    case mx::uint32:
      return ki::ToNode(env, a->item<uint32_t>());
    case mx::uint64:
      return ki::ToNode(env, a->item<uint64_t>());
    case mx::int8:
      return ki::ToNode(env, a->item<int8_t>());
    case mx::int16:
      return ki::ToNode(env, a->item<int16_t>());
    case mx::int32:
      return ki::ToNode(env, a->item<int32_t>());
    case mx::int64:
      return ki::ToNode(env, a->item<int64_t>());
    case mx::float16:
      return ki::ToNode(env, static_cast<float>(a->item<mx::float16_t>()));
    case mx::float32:
      return ki::ToNode(env, a->item<float>());
    case mx::bfloat16:
      return ki::ToNode(env, static_cast<float>(a->item<mx::bfloat16_t>()));
    case mx::complex64:
      // FIXME(zcbenz): Represent complex number in JS.
      return Undefined(env);
  }
}

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
          "complexfloating", mx::complexfloating,
          "floating", mx::floating,
          "inexact", mx::inexact,
          "signedinteger", mx::signedinteger,
          "unsignedinteger", mx::unsignedinteger,
          "integer", mx::integer,
          "number", mx::number,
          "generic", mx::generic);

  ki::Set(env, exports,
          "array", ki::Class<mx::array>());
}
