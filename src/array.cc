#include <set>
#include <stack>

#include "src/array.h"
#include "src/complex.h"
#include "src/indexing.h"
#include "src/ops.h"
#include "src/trees.h"

namespace {

// Implementation of the iterator.
class ArrayIterator {
 public:
  explicit ArrayIterator(mx::array x) : x_(std::move(x)) {
    // For small array using split for fast access.
    if (x_.shape(0) > 0 && x_.shape(0) < 10) {
      splits_ = mx::split(x_, x_.shape(0));
    }
  }

  napi_value Next(napi_env env) {
    napi_value result = ki::CreateObject(env);
    if (index_ >= x_.shape(0)) {
      // Returning {done: true} for out of range.
      ki::Set(env, result, "done", true);
    } else if (index_ >= 0 && index_ < splits_.size()) {
      // Use split results when available.
      ki::Set(env, result, "value", mx::squeeze(splits_[index_++], 0));
    } else {
      // Fallback to the C++ iterator, which uses slice.
      ki::Set(env, result, "value", *(x_.begin() + index_++));
    }
    return result;
  }

 private:
  mx::array x_;
  std::vector<mx::array> splits_;
  size_t index_ = 0;
};

// Create on heap if T is pointer, otherwise create on stack.
template<typename T, typename... Args>
inline T CreateInstance(Args&&... args) {
  if constexpr (std::is_pointer_v<T>)
    return new std::remove_pointer_t<T>(std::forward<Args>(args)...);
  else
    return typename T::value_type(std::forward<Args>(args)...);
}

// Get the shape of input nested array.
bool GetShape(napi_env env, napi_value value, mx::Shape* shape) {
  uint32_t length;
  if (napi_get_array_length(env, value, &length) != napi_ok)
    return false;
  shape->push_back(static_cast<int>(length));
  if (shape->back() == 0)
    return true;
  napi_value el;
  if (napi_get_element(env, value, 0, &el) != napi_ok)
    return false;
  if (ki::IsArray(env, el))
    return GetShape(env, el, shape);
  if (auto a = ki::FromNodeTo<mx::array*>(env, el); a) {
    for (int i = 0; i < a.value()->ndim(); ++i)
      shape->push_back(a.value()->shape(i));
    return true;
  }
  return true;
}

// The possible type of the elements of JS Array.
enum InputType {
  kBoolean,
  kUndefined,
  kNumber,
  kComplex,
};

// Validate whether the shape of input array is valid, and get information about
// the input.
bool ValidateInputArray(napi_env env,
                        napi_value value,
                        const mx::Shape& shape,
                        std::optional<InputType>* input_type,
                        size_t dim = 0) {
  if (dim >= shape.size()) {
    napi_throw_type_error(env, nullptr,
                          "Initialization encountered extra dimension.");
    return false;
  }

  // Input array should match the shape.
  uint32_t length;
  if (napi_get_array_length(env, value, &length) != napi_ok)
    return false;
  if (shape[dim] != length) {
    napi_throw_type_error(env, nullptr,
                          "Initialization encountered non-uniform length.");
    return false;
  }
  if (shape[dim] == 0)
    return true;

  // Iterate the elements to determine the dtype.
  for (uint32_t i = 0; i < length; ++i) {
    napi_value el;
    if (napi_get_element(env, value, i, &el) != napi_ok) {
      napi_throw_type_error(env, nullptr, "Unable to get array element");
      return false;
    }
    if (ki::IsArray(env, el)) {
      // Look into nested array.
      if (ValidateInputArray(env, el, shape, input_type, dim + 1))
        continue;
      else
        return false;
    }
    // Check the type of each element.
    napi_valuetype type;
    if (napi_typeof(env, el, &type) != napi_ok) {
      napi_throw_type_error(env, nullptr, "Unable to get array element type");
      return false;
    }
    if (type == napi_boolean) {
      *input_type = std::max(input_type->value_or(kBoolean), kBoolean);
    } else if (type == napi_number) {
      *input_type = std::max(input_type->value_or(kBoolean), kNumber);
    } else if (type == napi_object && IsComplexNumber(env, el)) {
      *input_type = std::max(input_type->value_or(kBoolean), kComplex);
    } else if (type == napi_undefined) {
      *input_type = std::max(input_type->value_or(kBoolean), kUndefined);
    } else {
      napi_throw_type_error(env, nullptr, "Unsupported array element type");
      return false;
    }
  }
  return true;
}

// Put all nested elements of array into a flat vector.
template<typename T>
bool FlattenArray(napi_env env, napi_value value, std::vector<T>* result) {
  uint32_t length;
  if (napi_get_array_length(env, value, &length) != napi_ok)
    return false;
  for (uint32_t i = 0; i < length; ++i) {
    napi_value el;
    if (napi_get_element(env, value, i, &el) != napi_ok)
      return false;
    if (ki::IsArray(env, el)) {
      FlattenArray(env, el, result);
    } else {
      std::optional<T> out = ki::FromNodeTo<T>(env, el);
      if (!out)
        return false;
      result->push_back(std::move(*out));
    }
  }
  return true;
}

// Convert JS Array to mx.array.
template<typename T>
T JsArrayToMxArray(napi_env env, napi_value value,
                   std::optional<mx::Dtype> dtype) {
  // Get array's shape, validate it and decide a proper dtype for it.
  mx::Shape shape;
  if (!GetShape(env, value, &shape))
    return T();
  // FIXME(zcbenz): Currently we assume the input array only includes primitive
  // types, we should support mx.array embedded in JS array.
  std::optional<InputType> largest_type;
  if (!ValidateInputArray(env, value, shape, &largest_type))
    return T();
  InputType element_type = largest_type.value_or(kNumber);
  if (element_type == kNumber) {
    std::vector<float> result;
    if (!FlattenArray(env, value, &result))
      return T();
    return CreateInstance<T>(result.begin(),
                             std::move(shape),
                             dtype.value_or(mx::float32));
  } else if (element_type == kBoolean) {
    std::vector<bool> result;
    if (!FlattenArray(env, value, &result))
      return T();
    return CreateInstance<T>(result.begin(),
                             std::move(shape),
                             dtype.value_or(mx::bool_));
  } else if (element_type == kComplex) {
    std::vector<std::complex<float>> result;
    if (!FlattenArray(env, value, &result))
      return T();
    return CreateInstance<T>(reinterpret_cast<mx::complex64_t*>(result.data()),
                             std::move(shape),
                             dtype.value_or(mx::complex64));
  } else if (element_type == kUndefined) {
    // Sparse array is treated as zeros.
    return CreateInstance<T>(mx::zeros(std::move(shape), mx::float32));
  } else {
    return T();
  }
}

// Convert JS TypedArray to mx.array.
template<typename T>
T JsTypedArrayToMxArray(napi_env env, napi_value value) {
  void* data;
  size_t length;
  napi_typedarray_type element_type;
  if (napi_get_typedarray_info(env, value, &element_type, &length, &data,
                               nullptr, nullptr) != napi_ok) {
    return T();
  }
  mx::Shape&& shape = { static_cast<int32_t>(length) };
  switch (element_type) {
    case napi_int8_array:
      return CreateInstance<T>(static_cast<int8_t*>(data), shape, mx::int8);
    case napi_uint8_array:
    case napi_uint8_clamped_array:
      return CreateInstance<T>(static_cast<uint8_t*>(data), shape, mx::uint8);
    case napi_int16_array:
      return CreateInstance<T>(static_cast<int16_t*>(data), shape, mx::int16);
    case napi_uint16_array:
      return CreateInstance<T>(static_cast<uint16_t*>(data), shape, mx::uint16);
    case napi_int32_array:
      return CreateInstance<T>(static_cast<int32_t*>(data), shape, mx::int32);
    case napi_uint32_array:
      return CreateInstance<T>(static_cast<uint32_t*>(data), shape, mx::uint32);
    case napi_float32_array:
      return CreateInstance<T>(static_cast<float*>(data), shape, mx::float32);
    case napi_float64_array:
      return CreateInstance<T>(static_cast<double*>(data), shape, mx::float64);
    default:
      napi_throw_type_error(env, nullptr, "Unsupported TypedArray type.");
      return T();
  }
}

// Implements the constructor of array, which can create on stack or heap.
template<typename T>
T CreateArray(napi_env env, napi_value value, std::optional<mx::Dtype> dtype) {
  napi_valuetype type;
  if (napi_typeof(env, value, &type) != napi_ok) {
    return T();
  }
  switch (type) {
    case napi_number:
      return CreateInstance<T>(ki::FromNodeTo<float>(env, value).value(),
                               dtype.value_or(mx::float32));
    case napi_boolean:
      return CreateInstance<T>(ki::FromNodeTo<bool>(env, value).value(),
                               dtype.value_or(mx::bool_));
    case napi_object: {
      // Convert JS array to mx.array.
      if (ki::IsArray(env, value))
        return JsArrayToMxArray<T>(env, value, dtype);
      if (ki::IsTypedArray(env, value))
        return JsTypedArrayToMxArray<T>(env, value);
      // Test if it is an mx.array instance.
      auto a = ki::FromNodeTo<mx::array*>(env, value);
      if (a) {
        return CreateInstance<T>(
            mx::astype(*a.value(), dtype.value_or(a.value()->dtype())));
      }
      // Test complex number.
      auto c = ki::FromNodeTo<std::complex<float>>(env, value);
      if (c) {
        return CreateInstance<T>(*c, dtype.value_or(mx::complex64));
      }
      [[fallthrough]];
    }
    default: {
      napi_throw_type_error(env, nullptr, "Unsupported array type.");
      return T();
    }
  }
}

// Implementation of the length property.
int Length(mx::array* a, napi_env env) {
  if (a->ndim() == 0) {
    napi_throw_type_error(env, nullptr, "0-dimensional array has no length.");
    return 0;
  }
  return a->shape(0);
}

// Return the ArrayAt helper.
ArrayAt* At(mx::array* a, ki::Arguments* args) {
  if (args->RemainingsLength() == 1) {
    auto index = args->GetNext<ArrayIndex>();
    if (!index) {
      args->ThrowError("Index");
      return nullptr;
    }
    return new ArrayAt(*a, std::move(*index));
  }
  ArrayIndices indices;
  if (!ReadArgs(args, &indices)) {
    args->ThrowError("Index");
    return nullptr;
  }
  return new ArrayAt(*a, std::move(indices));
}

// Helper to get the array data with type.
template<typename F>
auto VisitArrayData(F&& visitor, mx::array* a) {
  switch (a->dtype()) {
    case mx::bool_:
      return visitor(a->data<bool>());
    case mx::uint8:
      return visitor(a->data<uint8_t>());
    case mx::uint16:
      return visitor(a->data<uint16_t>());
    case mx::uint32:
      return visitor(a->data<uint32_t>());
    case mx::uint64:
      return visitor(a->data<uint64_t>());
    case mx::int8:
      return visitor(a->data<int8_t>());
    case mx::int16:
      return visitor(a->data<int16_t>());
    case mx::int32:
      return visitor(a->data<int32_t>());
    case mx::int64:
      return visitor(a->data<int64_t>());
    case mx::float16:
      return visitor(a->data<mx::float16_t>());
    case mx::float32:
      return visitor(a->data<float>());
    case mx::float64:
      return visitor(a->data<double>());
    case mx::bfloat16:
      return visitor(a->data<mx::bfloat16_t>());
    case mx::complex64:
      return visitor(a->data<std::complex<float>>());
  }
}

// Convert the array to scalar.
napi_value Item(mx::array* a, napi_env env) {
  if (a->size() != 1) {
    napi_throw_error(env, nullptr,
                     "item() can only be called on arrays of size 1.");
    return nullptr;
  }
  a->eval();
  return VisitArrayData([env](auto* data) {
    return ki::ToNodeValue(env, *data);
  }, a);
}

// Convert mx::array to JS array.
template<typename T>
napi_value MxArrayToJsArray(napi_env env,
                            const mx::array& a,
                            T* data,
                            size_t index = 0,
                            int dim = 0) {
  napi_value ret;
  if (napi_create_array_with_length(env, a.shape(dim), &ret) != napi_ok)
    return nullptr;
  size_t stride = a.strides()[dim];
  for (size_t i = 0; i < a.shape(dim); ++i) {
    if (dim == a.ndim() - 1) {
      // The last dimension only has scalars.
      napi_set_element(env, ret, i, ki::ToNodeValue(env, data[index]));
    } else {
      napi_set_element(env, ret, i,
                       MxArrayToJsArray(env, a, data, index, dim + 1));
    }
    index += stride;
  }
  return ret;
}

// Implementation of the tolist method.
napi_value ToList(mx::array* a, napi_env env) {
  if (a->ndim() == 0)
    return Item(a, env);
  a->eval();
  return VisitArrayData([env, a](auto* data) {
    return MxArrayToJsArray(env, *a, data);
  }, a);
}

// Convert the data to TypedArray.
napi_value ToTypedArray(mx::array* a, napi_env env) {
  if (a->ndim() != 1) {
    napi_throw_type_error(env, nullptr, "Array must have only one dimension.");
    return nullptr;
  }
  napi_typedarray_type type;
  switch (a->dtype()) {
    case mx::uint8:  type = napi_uint8_array; break;
    case mx::uint16: type = napi_uint16_array; break;
    case mx::uint32: type = napi_uint32_array; break;
    case mx::int8:  type = napi_int8_array; break;
    case mx::int16: type = napi_int16_array; break;
    case mx::int32: type = napi_int32_array; break;
    case mx::float32: type = napi_float32_array; break;
    case mx::float64: type = napi_float64_array; break;
    default:
      napi_throw_type_error(env, nullptr, "No matching TypedArray for dtype.");
      return nullptr;
  }
  a->eval();
  // Create a ArrayBuffer that stores a reference to array's data.
  using DataType = std::shared_ptr<mx::array::Data>;
  napi_value buffer;
  if (napi_create_external_arraybuffer(env,
                                       a->data<uint8_t>(), a->nbytes(),
                                       [](napi_env env, void*, void* hint) {
                                          delete static_cast<DataType*>(hint);
                                       },
                                       new DataType(a->data_shared_ptr()),
                                       &buffer) != napi_ok) {
    napi_throw_error(env, nullptr, "Unable to create ArrayBuffer.");
    return nullptr;
  }
  // Create view on the ArrayBuffer.
  napi_value buffer_view;
  if (napi_create_typedarray(env, type, a->size(), buffer, 0,
                             &buffer_view) != napi_ok) {
    napi_throw_error(env, nullptr, "Unable to create TypedArray.");
    return nullptr;
  }
  return buffer_view;
}

// Get the Symbol.iterator.
napi_value SymbolIterator(napi_env env) {
  napi_value result;
  if (ki::Get(env, ki::Global(env), "Symbol", &result) &&
      ki::Get(env, result, "iterator", &result)) {
    return result;
  }
  return nullptr;
}

// Implement the [Symbol.iterator] method.
ArrayIterator* CreateArrayIterator(mx::array* a) {
  return new ArrayIterator(*a);
}

// Implement the finfo function.
napi_value FInfo(napi_env env, mx::Dtype dtype) {
  mx::finfo info = mx::finfo(dtype);
  napi_value result = ki::CreateObject(env);
  ki::Set(env, result, "dtype", info.dtype,
                       "min", info.min,
                       "max", info.max);
  return result;
}

// Store array pointers allocated during a tidy call.
std::stack<std::set<mx::array*>> g_tidy_arrays;

// Release all array pointers allocated during the call.
napi_value Tidy(napi_env env, std::function<napi_value()> func) {
  // Push a new set to stack.
  g_tidy_arrays.push(std::set<mx::array*>());
  auto& top = g_tidy_arrays.top();
  return AwaitFunction(
      env, std::move(func),
      [&top](napi_env env, napi_value result) {
        // Exclude the arrays in result from the stack.
        TreeVisit(env, result, [&top](napi_env env, napi_value value) {
          if (auto a = ki::FromNodeTo<mx::array*>(env, value); a)
            top.erase(*a);
          return napi_value();
        });
        // Clear the arrays in the stack.
        ki::InstanceData* instance_data = ki::InstanceData::Get(env);
        for (mx::array* a : top) {
          // The arary might be in 3 states:
          // 1. Its JS object is well alive.
          // 2. The JS object has been fully GCed.
          // 3. The JS object is marked as dead, but the finalizer has not run.
          // We have to unbind the JS object in 1, and only delete array in 1
          // and 3.
          napi_value value;
          if (instance_data->GetWrapper<mx::array>(a, &value))
            napi_remove_wrap(env, value, nullptr);
          if (instance_data->DeleteWrapper<mx::array>(a))
            delete a;
        }
        return result;
      },
      [](napi_env env) {
        // Always pop even when error happened.
        g_tidy_arrays.pop();
      });
}

// Dispose all arrays found in the tree.
void Dispose(const ki::Arguments& args) {
  ki::InstanceData* instance_data = ki::InstanceData::Get(args.Env());
  for (size_t i = 0; i < args.Length(); ++i) {
    TreeVisit(args.Env(), args[i],
              [instance_data](napi_env env, napi_value value) {
                if (auto a = ki::FromNodeTo<mx::array*>(env, value); a) {
                  napi_remove_wrap(env, value, nullptr);
                  instance_data->DeleteWrapper<mx::array>(a.value());
                  delete a.value();
                }
                return napi_value();
              });
  }
}

// Return how many wrappers are there, used for debugging leaks.
size_t GetWrappersCount(napi_env env) {
  return ki::InstanceData::Get(env)->GetWrappersCount();
}

}  // namespace

namespace ki {

// Allow passing Dtype to JS directly, no memory management involved as they are
// static globals.
template<>
struct TypeBridge<mx::Dtype> {
  static inline mx::Dtype* Wrap(mx::Dtype* ptr) {
    return ptr;
  }
  static inline void Finalize(mx::Dtype* ptr) {
  }
};

// static
void Type<mx::Dtype>::Define(napi_env env,
                             napi_value constructor,
                             napi_value prototype) {
  DefineProperties(env, prototype,
                   Property("size", Getter(&mx::Dtype::size)));
  DefineToString<mx::Dtype>(env, prototype);
}

// static
napi_status Type<mx::Dtype>::ToNode(napi_env env,
                                    const mx::Dtype& value,
                                    napi_value* result) {
  // Make sure the same Dtype ends up converting to the same object so it is
  // possible to compare Dtype in JS.
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
  if (value == mx::float64)
    return ConvertToNode(env, &mx::float64, result);
  if (value == mx::bfloat16)
    return ConvertToNode(env, &mx::bfloat16, result);
  if (value == mx::complex64)
    return ConvertToNode(env, &mx::complex64, result);
  return napi_generic_failure;
}

// static
std::optional<mx::Dtype> Type<mx::Dtype>::FromNode(napi_env env,
                                                   napi_value value) {
  return AllowPassByValue<mx::Dtype>::FromNode(env, value);
}

// static
napi_status Type<mx::Dtype::Category>::ToNode(
    napi_env env, mx::Dtype::Category type, napi_value* result) {
  return ConvertToNode(env, static_cast<int>(type), result);
}

// static
std::optional<mx::Dtype::Category> Type<mx::Dtype::Category>::FromNode(
    napi_env env, napi_value value) {
  std::optional<int> type = FromNodeTo<int>(env, value);
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

template<>
struct TypeBridge<mx::array> {
  static inline mx::array* Wrap(mx::array* ptr) {
    if (!g_tidy_arrays.empty())
      g_tidy_arrays.top().insert(ptr);
    return ptr;
  }
  static inline void Finalize(mx::array* ptr) {
    // Every array pointer passed to JS is managed by us.
    delete ptr;
  }
};

// static
mx::array* Type<mx::array>::Constructor(napi_env env,
                                        napi_value value,
                                        std::optional<mx::Dtype> dtype) {
  return CreateArray<mx::array*>(env, value, std::move(dtype));
}

// static
void Type<mx::array>::Define(napi_env env,
                             napi_value constructor,
                             napi_value prototype) {
  // Disambiguate the 2 overloads of shape().
  using shape_fun = const mx::Shape& (mx::array::*)() const;
  shape_fun shape = &mx::array::shape;
  // Disambiguate the 3 overloads of transpose().
  using t_fun = mx::array (*)(const mx::array&, mx::StreamOrDevice);
  t_fun t = &mx::transpose;
  // Define array's properties.
  DefineProperties(env, prototype,
                   Property("length", Getter(MemberFunction(&Length))),
                   Property("size", Getter(&mx::array::size)),
                   Property("ndim", Getter(&mx::array::ndim)),
                   Property("itemsize", Getter(&mx::array::itemsize)),
                   Property("nbytes", Getter(&mx::array::nbytes)),
                   Property("shape", Getter(shape)),
                   Property("dtype", Getter(&mx::array::dtype)),
                   Property("T", Getter(MemberFunction(t))));
  // Define array's methods.
  Set(env, prototype,
      "at", MemberFunction(&At),
      "item", MemberFunction(&Item),
      "tolist", MemberFunction(&ToList),
      "toTypedArray", MemberFunction(&ToTypedArray),
      "astype", MemberFunction(&mx::astype),
      "flatten", MemberFunction(&ops::Flatten),
      "reshape", MemberFunction(&ops::Reshape),
      "squeeze", MemberFunction(&ops::Squeeze),
      "abs", MemberFunction(&mx::abs),
      "square", MemberFunction(&mx::square),
      "sqrt", MemberFunction(&mx::sqrt),
      "rsqrt", MemberFunction(&mx::rsqrt),
      "reciprocal", MemberFunction(&mx::reciprocal),
      "power", MemberFunction(&mx::power),
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
      "swapaxes", MemberFunction(&mx::swapaxes),
      "transpose", MemberFunction(&ops::Transpose),
      "sum", MemberFunction(DimOpWrapper(&mx::sum)),
      "prod", MemberFunction(DimOpWrapper(&mx::prod)),
      "min", MemberFunction(DimOpWrapper(&mx::min)),
      "max", MemberFunction(DimOpWrapper(&mx::max)),
      "logcumsumexp", MemberFunction(CumOpWrapper(&mx::logcumsumexp)),
      "logsumexp", MemberFunction(DimOpWrapper(&mx::logsumexp)),
      "mean", MemberFunction(DimOpWrapper(&mx::mean)),
      "std", MemberFunction(&ops::Std),
      "variance", MemberFunction(&ops::Var),
      "split", MemberFunction(&ops::Split),
      "argmin", MemberFunction(&ops::ArgMin),
      "argmax", MemberFunction(&ops::ArgMax),
      "cumsum", MemberFunction(CumOpWrapper(&mx::cumsum)),
      "cumprod", MemberFunction(CumOpWrapper(&mx::cumprod)),
      "cummax", MemberFunction(CumOpWrapper(&mx::cummax)),
      "cummin", MemberFunction(CumOpWrapper(&mx::cummin)),
      "round", MemberFunction(&ops::Round),
      "diagonal", MemberFunction(&ops::Diagonal),
      "diag", MemberFunction(&ops::Diag),
      "conj", MemberFunction(&mx::conjugate),
      "view", MemberFunction(&mx::view),
      "index", MemberFunction(&Index),
      "indexPut_", MemberFunction(&IndexPut),
      SymbolIterator(env), MemberFunction(&CreateArrayIterator));
  DefineToString<mx::array>(env, prototype);
}

// static
std::optional<mx::array> Type<mx::array>::FromNode(napi_env env,
                                                   napi_value value) {
  // First try to get the array instance directly.
  auto a = AllowPassByValue<mx::array>::FromNode(env, value);
  if (a)
    return a;
  // Fallback to creating an array.
  return CreateArray<std::optional<mx::array>>(env, value, std::nullopt);
}

template<>
struct TypeBridge<ArrayAt> {
  static inline ArrayAt* Wrap(ArrayAt* ptr) {
    return ptr;
  }
  static inline void Finalize(ArrayAt* ptr) {
    delete ptr;
  }
};

template<>
struct Type<ArrayAt> {
  static constexpr const char* name = "ArrayAt";
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype) {
    Set(env, prototype,
        "add", &ArrayAt::Add,
        "subtract", &ArrayAt::Subtract,
        "multiply", &ArrayAt::Multiply,
        "divide", &ArrayAt::Divide,
        "maximum", &ArrayAt::Maximum,
        "minimum", &ArrayAt::Minimum);
  }
};

template<>
struct TypeBridge<ArrayIterator> {
  static inline ArrayIterator* Wrap(ArrayIterator* ptr) {
    return ptr;
  }
  static inline void Finalize(ArrayIterator* ptr) {
    delete ptr;
  }
};

template<>
struct Type<ArrayIterator> {
  static constexpr const char* name = "ArrayIterator";
  static void Define(napi_env env,
                     napi_value constructor,
                     napi_value prototype) {
    Set(env, prototype,
        "next", &ArrayIterator::Next);
  }
};

}  // namespace ki

void InitArray(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "Dtype", ki::Class<mx::Dtype>(),
          "bool", &mx::bool_,
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
          "float64", &mx::float64,
          "bfloat16", &mx::bfloat16,
          "complex64", &mx::complex64);

  ki::Set(env, exports,
          "DtypeCategory", ki::Class<mx::Dtype::Category>(),
          "complexfloating", mx::complexfloating,
          "floating", mx::floating,
          "inexact", mx::inexact,
          "signedinteger", mx::signedinteger,
          "unsignedinteger", mx::unsignedinteger,
          "integer", mx::integer,
          "number", mx::number,
          "generic", mx::generic);

  ki::Set(env, exports,
          "array", ki::Class<mx::array>(),
          "finfo", &FInfo);

  ki::Set(env, exports,
          "tidy", &Tidy,
          "dispose", &Dispose,
          "getWrappersCount", &GetWrappersCount);
}
