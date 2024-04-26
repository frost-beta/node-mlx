#include "src/array.h"
#include "src/indexing.h"

#include <numeric>

namespace {

// Whether the slice is filled with null, which means a copy of array.
inline bool IsSliceNone(const Slice& slice) {
  return !slice.start && !slice.stop && !slice.step;
}

// Get slice values depending on the array's |length|.
inline void ReadSlice(const Slice& slice,
                      int length,
                      int* start,
                      int* stop,
                      int* step) {
  // Following numpy's convention:
  //    Assume n is the number of elements in the dimension being sliced.
  //    Then, if i is not given it defaults to 0 for k > 0 and n - 1 for
  //    k < 0 . If j is not given it defaults to n for k > 0 and -n-1 for
  //    k < 0 . If k is not given it defaults to 1
  *step = slice.step.value_or(1);
  *start = slice.start.value_or(*step < 0 ? length - 1 : 0);
  *stop = slice.stop.value_or(*step < 0 ? -length - 1 : length);
}

// Convert negative index to positive array.
inline mx::array GetIntIndex(int index, int length) {
  if (index < 0)
    index += length;
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

// The implementation comes from mlx_gather_nd.
mx::array GatherNDimensional(mx::array a,
                             const std::vector<ArrayIndex>& indices,
                             bool gather_first,
                             size_t* max_dims) {
  *max_dims = 0;
  std::vector<mx::array> gather_indices;
  std::vector<bool> is_slice(indices.size(), false);
  size_t num_slices = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    const ArrayIndex& index = indices[i];
    if (std::holds_alternative<Slice>(index)) {
      int start, stop, step;
      ReadSlice(std::get<Slice>(index), a.shape(i), &start, &stop, &step);

      start = start < 0 ? start + a.shape(i) : start;
      stop = stop < 0 ? stop + a.shape(i) : stop;

      gather_indices.push_back(mx::arange(start, stop, step, mx::uint32));
      num_slices++;
      is_slice[i] = true;
    } else if (std::holds_alternative<int>(index)) {
      gather_indices.push_back(GetIntIndex(std::get<int>(index), a.shape(i)));
    } else if (std::holds_alternative<mx::array*>(index)) {
      mx::array* arr = std::get<mx::array*>(index);
      *max_dims = std::max(arr->ndim(), *max_dims);
      gather_indices.push_back(*arr);
    }
  }

  if (gather_first) {
    size_t slice_index = 0;
    for (size_t i = 0; i < gather_indices.size(); i++) {
      if (is_slice[i]) {
        std::vector<int> index_shape(*max_dims + num_slices, 1);
        index_shape[*max_dims + slice_index] = gather_indices[i].shape(0);
        gather_indices[i] = mx::reshape(gather_indices[i], index_shape);
        slice_index++;
      } else {
        std::vector<int> index_shape = gather_indices[i].shape();
        index_shape.insert(index_shape.end(), num_slices, 1);
        gather_indices[i] = mx::reshape(gather_indices[i], index_shape);
      }
    }
  } else {
    for (size_t i = 0; i < gather_indices.size(); i++) {
      if (i < num_slices) {
        std::vector<int> index_shape(*max_dims + num_slices, 1);
        index_shape[i] = gather_indices[i].shape(0);
        gather_indices[i] = mx::reshape(gather_indices[i], index_shape);
      }
    }
  }

  std::vector<int> axes(indices.size());
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<int> slice_sizes = a.shape();
  std::fill(slice_sizes.begin(), slice_sizes.begin() + indices.size(), 1);
  mx::array gathered = mx::gather(std::move(a), std::move(gather_indices),
                                  std::move(axes), std::move(slice_sizes));

  std::vector<int> out_shape;
  out_shape.insert(
      out_shape.end(),
      gathered.shape().begin(),
      gathered.shape().begin() + *max_dims + num_slices);
  out_shape.insert(
      out_shape.end(),
      gathered.shape().begin() + *max_dims + num_slices + indices.size(),
      gathered.shape().end());
  return mx::reshape(std::move(gathered), std::move(out_shape));
}

// The implementation comes from mlx_expand_ellipsis.
std::pair<size_t, std::vector<ArrayIndex>> ExpandEllipsis(
    const std::vector<int>& shape,
    const std::vector<ArrayIndex>& entries) {
  std::vector<ArrayIndex> indices;
  std::vector<ArrayIndex> r_indices;

  size_t non_none_indices_before = 0;
  size_t non_none_indices_after = 0;
  bool has_ellipsis = false;

  size_t i = 0;
  for (; i < entries.size(); i++) {
    const ArrayIndex& index = entries[i];
    if (!std::holds_alternative<Ellipsis>(index)) {
      if (!std::holds_alternative<std::monostate>(index))
        non_none_indices_before++;
      indices.push_back(index);
    } else {
      has_ellipsis = true;
      break;
    }
  }

  for (size_t j = entries.size() - 1; j > i; j--) {
    const ArrayIndex& index = entries[j];
    if (std::holds_alternative<Ellipsis>(index)) {
      throw std::invalid_argument(
          "An index can only have a single ellipsis ('...')");
    }
    r_indices.push_back(index);
    if (!std::holds_alternative<std::monostate>(index))
      non_none_indices_after++;
  }

  size_t non_none_indices = non_none_indices_before + non_none_indices_after;

  if (has_ellipsis) {
    for (size_t axis = non_none_indices_before;
         axis < shape.size() - non_none_indices_after;
         axis++) {
      indices.push_back(Slice{0, shape[axis], 1});
      non_none_indices++;
    }
  }

  indices.insert(indices.end(), r_indices.rbegin(), r_indices.rend());
  return std::make_pair(non_none_indices, std::move(indices));
}

// The implementation comes from mlx_get_item_nd.
mx::array IndexNDimensional(const mx::array* a,
                            std::vector<ArrayIndex> entries) {
  if (entries.size() == 0)
    return *a;

  auto [non_none_indices, indices] = ExpandEllipsis(a->shape(), entries);
  if (non_none_indices > a->ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << a->ndim() << "dimensions.";
    throw std::invalid_argument(msg.str());
  }

  mx::array gathered = *a;
  std::vector<ArrayIndex> remaining_indices;
  bool have_array = false;
  {
    bool have_non_array = false;
    bool gather_first = false;
    for (const ArrayIndex& index : indices) {
      if (std::holds_alternative<mx::array*>(index) ||
          std::holds_alternative<int>(index)) {
        if (have_array && have_non_array) {
          gather_first = true;
          break;
        }
        have_array = true;
      } else {
        have_non_array |= have_array;
      }
    }

    bool have_array_instance = false;
    for (const ArrayIndex& index : indices) {
      if (std::holds_alternative<mx::array*>(index)) {
        have_array_instance = true;
        break;
      }
    }
    have_array &= have_array_instance;

    if (have_array) {
      size_t last_array;
      for (last_array = indices.size() - 1; last_array >= 0; last_array--) {
        const ArrayIndex& index = indices[last_array];
        if (std::holds_alternative<mx::array*>(index) ||
            std::holds_alternative<int>(index)) {
          break;
        }
      }

      std::vector<ArrayIndex> gather_indices;
      for (size_t i = 0; i <= last_array; i++) {
        const ArrayIndex& index = indices[i];
        if (!std::holds_alternative<std::monostate>(index)) {
          gather_indices.push_back(index);
        }
      }

      size_t max_dims;
      gathered = GatherNDimensional(std::move(gathered), gather_indices,
                                    gather_first, &max_dims);

      if (gather_first) {
        for (size_t i = 0; i < max_dims; i++) {
          remaining_indices.push_back(Slice{});
        }
        for (size_t i = 0; i < last_array; i++) {
          const ArrayIndex& index = indices[i];
          if (std::holds_alternative<std::monostate>(index))
            remaining_indices.push_back(index);
          else if (std::holds_alternative<Slice>(index))
            remaining_indices.push_back(Slice{});
        }
        for (size_t i = last_array + 1; i < indices.size(); i++) {
          remaining_indices.push_back(indices[i]);
        }
      } else {
        for (size_t i = 0; i < indices.size(); i++) {
          const ArrayIndex& index = indices[i];
          if (std::holds_alternative<mx::array*>(index) ||
              std::holds_alternative<int>(index)) {
            break;
          } else if (std::holds_alternative<std::monostate>(index)) {
            remaining_indices.push_back(index);
          } else {
            remaining_indices.push_back(Slice{});
          }
        }
        for (size_t i = 0; i < max_dims; i++) {
          remaining_indices.push_back(Slice{});
        }
        for (size_t i = last_array + 1; i < indices.size(); i++) {
          remaining_indices.push_back(indices[i]);
        }
      }
    }
  }
  if (have_array && remaining_indices.empty()) {
    return gathered;
  }
  if (remaining_indices.empty()) {
    remaining_indices = indices;
  }

  bool squeeze_needed = false;
  bool unsqueeze_needed = false;
  {
    std::vector<int> starts(gathered.ndim(), 0);
    std::vector<int> stops(gathered.shape());
    std::vector<int> steps(gathered.ndim(), 1);
    size_t axis = 0;
    for (const ArrayIndex& index : remaining_indices) {
      if (!std::holds_alternative<std::monostate>(index)) {
        if (!have_array && std::holds_alternative<int>(index)) {
          int start = std::get<int>(index);
          start = start < 0 ? start + gathered.shape(axis) : start;
          starts[axis] = start;
          stops[axis] = start + 1;
          squeeze_needed = true;
        } else {
          ReadSlice(std::get<Slice>(index), stops[axis],
                    &starts[axis], &stops[axis], &steps[axis]);
        }
        axis++;
      } else {
        unsqueeze_needed = true;
      }
    }
    gathered = mx::slice(std::move(gathered), starts, stops, steps);
  }

  if (unsqueeze_needed || squeeze_needed) {
    std::vector<int> out_shape;
    size_t axis = 0;
    for (const ArrayIndex& index : remaining_indices) {
      if (unsqueeze_needed && std::holds_alternative<std::monostate>(index)) {
        out_shape.push_back(1);
      } else if (squeeze_needed && std::holds_alternative<int>(index)) {
        axis++;
      } else {
        out_shape.push_back(gathered.shape(axis++));
      }
    }

    out_shape.insert(out_shape.end(),
                     gathered.shape().begin() + axis, gathered.shape().end());
    gathered = mx::reshape(std::move(gathered), std::move(out_shape));
  }

  return gathered;
}

// Index an array with only one index.
mx::array IndexOne(mx::array* a, ArrayIndex index) {
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

}  // namespace

mx::array Index(mx::array* a, ki::Arguments* args) {
  if (args->Length() == 1) {
    auto index = args->GetNext<ArrayIndex>();
    if (!index) {
      args->ThrowError("Index");
      return *a;
    }
    return IndexOne(a, std::move(*index));
  }
  ArrayIndices indices;
  if (ReadArgs(args, &indices) != args->Length()) {
    args->ThrowError("Index");
    return *a;
  }
  return IndexNDimensional(a, std::move(indices));
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
      "start", value.start,
      "stop", value.stop,
      "step", value.step);
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
