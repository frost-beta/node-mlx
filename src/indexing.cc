#include "src/indexing.h"

namespace {

// Whether the slice is filled with null, which means a copy of array.
inline bool IsSliceNone(const Slice& slice) {
  return !slice.start && !slice.stop && !slice.step;
}

// Get slice values depending on the array's |shape|.
inline void ReadSlice(const Slice& slice,
                      int shape,
                      int* start,
                      int* stop,
                      int* step) {
  // Following numpy's convention:
  //    Assume n is the number of elements in the dimension being sliced.
  //    Then, if i is not given it defaults to 0 for k > 0 and n - 1 for
  //    k < 0 . If j is not given it defaults to n for k > 0 and -n-1 for
  //    k < 0 . If k is not given it defaults to 1
  *step = slice.step.value_or(1);
  *start = slice.start.value_or(*step < 0 ? shape - 1 : 0);
  *stop = slice.stop.value_or(*step < 0 ? -shape - 1 : shape);
}

// Convert negative index to positive array.
inline mx::array GetIntIndex(int index, int shape) {
  if (index < 0)
    index += shape;
  return mx::array(index, mx::uint32);
}

// Index an array with |slice|.
mx::array IndexSlice(mx::array* a, const Slice& slice) {
  if (IsSliceNone(slice))
    return *a;
  std::vector<int> starts(a->ndim(), 0);
  std::vector<int> stops(a->shape());
  std::vector<int> steps(a->ndim(), 1);
  ReadSlice(slice, a->shape(0), &starts[0], &stops[0], &steps[0]);
  return mx::slice(*a, std::move(starts), std::move(stops), std::move(steps));
}

}  // namespace

mx::array Index(mx::array* a, ArrayIndex index) {
  if (std::holds_alternative<std::monostate>(index)) {
    std::vector<int> shape = { 1 };
    shape.insert(shape.end(), a->shape().begin(), a->shape().end());
    return mx::reshape(*a, std::move(shape));
  }
  if (std::holds_alternative<Ellipsis>(index)) {
    return *a;
  }
  if (a->ndim() == 0) {
    throw std::invalid_argument("too many indices for array: "
                                "array is 0-dimensional");
  }
  if (auto s = std::get_if<Slice>(&index); s) {
    return IndexSlice(a, *s);
  }
  if (auto p = std::get_if<mx::array*>(&index); p) {
    if ((*p)->dtype() == mx::bool_)
      throw std::invalid_argument("boolean indices are not yet supported.");
    return mx::take(*a, **p, 0);
  }
  if (auto i = std::get_if<int>(&index); i) {
    return mx::take(*a, GetIntIndex(*i, a->shape(0)), 0);
  }
  throw std::invalid_argument("Cannot index mlx array using the given type.");
}

namespace ki {

// static
napi_status Type<Slice>::ToNode(napi_env env,
                                const Slice& value,
                                napi_value* result) {
  napi_status s = napi_create_object(env, result);
  if (s != napi_ok)
    return s;
  Set(env, *result,
      "start", *value.start,
      "stop", *value.stop,
      "step", *value.step);
  return napi_ok;
}

// static
std::optional<Slice> Type<Slice>::FromNode(napi_env env,
                                           napi_value value) {
  if (!IsType(env, value, napi_object))
    return std::nullopt;
  Slice result;
  if (!Get(env, value,
           "start", &result.start,
           "stop", &result.stop,
           "step", &result.step)) {
    return std::nullopt;
  }
  return result;
}

// static
napi_status Type<Ellipsis>::ToNode(napi_env env,
                                   const Ellipsis& value,
                                   napi_value* result) {
  return ConvertToNode(env, "...", result);
}

// static
std::optional<Ellipsis> Type<Ellipsis>::FromNode(napi_env env,
                                                 napi_value value) {
  auto str = FromNodeTo<std::string>(env, value);
  if (str && *str == "...")
    return Ellipsis();
  return std::nullopt;
}

}  // namespace ki

Slice CreateSlice(std::optional<int> start,
                  std::optional<int> stop,
                  std::optional<int> step) {
  return {start, stop, step};
}

void InitIndexing(napi_env env, napi_value exports) {
  ki::Set(env, exports, "Slice", &CreateSlice);
}
