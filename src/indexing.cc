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
                      mx::ShapeElem* start,
                      mx::ShapeElem* stop,
                      mx::ShapeElem* step) {
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
mx::array IndexSlice(const mx::array* a, const Slice& slice) {
  if (IsSliceNone(slice))
    return *a;
  mx::Shape starts(a->ndim(), 0);
  mx::Shape stops(a->shape());
  mx::Shape steps(a->ndim(), 1);
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
        mx::Shape index_shape(*max_dims + num_slices, 1);
        index_shape[*max_dims + slice_index] = gather_indices[i].shape(0);
        gather_indices[i] = mx::reshape(gather_indices[i],
                                        std::move(index_shape));
        slice_index++;
      } else {
        mx::Shape index_shape = gather_indices[i].shape();
        index_shape.insert(index_shape.end(), num_slices, 1);
        gather_indices[i] = mx::reshape(gather_indices[i],
                                        std::move(index_shape));
      }
    }
  } else {
    for (size_t i = 0; i < gather_indices.size(); i++) {
      if (i < num_slices) {
        mx::Shape index_shape(*max_dims + num_slices, 1);
        index_shape[i] = gather_indices[i].shape(0);
        gather_indices[i] = mx::reshape(gather_indices[i],
                                        std::move(index_shape));
      }
    }
  }

  std::vector<int> axes(indices.size());
  std::iota(axes.begin(), axes.end(), 0);
  mx::Shape slice_sizes = a.shape();
  std::fill(slice_sizes.begin(), slice_sizes.begin() + indices.size(), 1);
  mx::array gathered = mx::gather(std::move(a), std::move(gather_indices),
                                  std::move(axes), std::move(slice_sizes));

  for (auto& ax : axes) {
    ax += *max_dims + num_slices;
  }
  return mx::squeeze(std::move(gathered), std::move(axes));
}

// The implementation comes from mlx_expand_ellipsis.
std::pair<size_t, std::vector<ArrayIndex>> ExpandEllipsis(
    const mx::Shape& shape,
    std::vector<ArrayIndex> entries) {
  if (entries.size() == 0)
    return {0, {}};

  std::vector<ArrayIndex> indices;
  std::vector<ArrayIndex> r_indices;

  size_t non_none_indices_before = 0;
  size_t non_none_indices_after = 0;
  bool has_ellipsis = false;

  size_t i = 0;
  for (; i < entries.size(); i++) {
    ArrayIndex& index = entries[i];
    if (!std::holds_alternative<Ellipsis>(index)) {
      if (!std::holds_alternative<std::monostate>(index))
        non_none_indices_before++;
      indices.push_back(std::move(index));
    } else {
      has_ellipsis = true;
      break;
    }
  }

  for (size_t j = entries.size() - 1; j > i; j--) {
    ArrayIndex& index = entries[j];
    if (std::holds_alternative<Ellipsis>(index)) {
      throw std::invalid_argument(
          "An index can only have a single ellipsis ('...')");
    }
    if (!std::holds_alternative<std::monostate>(index))
      non_none_indices_after++;
    r_indices.push_back(std::move(index));
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

  indices.insert(indices.end(),
                 std::make_move_iterator(r_indices.rbegin()),
                 std::make_move_iterator(r_indices.rend()));
  return std::make_pair(non_none_indices, std::move(indices));
}

// Index an array with multiple indices.
// The implementation comes from mlx_get_item_nd.
mx::array IndexNDimensional(const mx::array* a,
                            std::vector<ArrayIndex> entries) {
  if (entries.size() == 0)
    return *a;

  auto [non_none_indices, indices] = ExpandEllipsis(a->shape(),
                                                    std::move(entries));
  if (non_none_indices > a->ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << a->ndim() << " dimensions.";
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
    mx::Shape starts(gathered.ndim(), 0);
    mx::Shape stops(gathered.shape());
    mx::Shape steps(gathered.ndim(), 1);
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
    std::vector<int> squeeze_axes;
    std::vector<int> unsqueeze_axes;
    for (int axis = 0; axis < remaining_indices.size(); ++axis) {
      ArrayIndex& index = remaining_indices[axis];
      if (unsqueeze_needed && std::holds_alternative<std::monostate>(index)) {
        unsqueeze_axes.push_back(axis - squeeze_axes.size());
      } else if (squeeze_needed && std::holds_alternative<int>(index)) {
        squeeze_axes.push_back(axis - unsqueeze_axes.size());
      }
    }
    if (!squeeze_axes.empty()) {
      gathered = mx::squeeze(std::move(gathered), std::move(squeeze_axes));
    }
    if (!unsqueeze_axes.empty()) {
      gathered = mx::expand_dims(std::move(gathered),
                                 std::move(unsqueeze_axes));
    }
  }

  return gathered;
}

// Index an array with only one index.
// Modified from mlx_get_item.
mx::array IndexOne(const mx::array* a, ArrayIndex index) {
  if (std::holds_alternative<std::monostate>(index)) {
    mx::Shape shape = { 1 };
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

// Return a new shape that removes begining dimensions with size 1.
mx::Shape GetUpShape(const mx::array& a) {
  size_t s = 0;
  while (s < a.ndim() && a.shape(s) == 1)
    s++;
  return mx::Shape(a.shape().begin() + s, a.shape().end());
}

using ScatterResult = std::tuple<std::vector<mx::array>,
                                 mx::array,
                                 std::vector<int>>;

// The implementation comes from mlx_scatter_args_int.
ScatterResult ScatterArgsInt(const mx::array* a,
                             int index,
                             mx::array update) {
  mx::Shape up_shape = GetUpShape(update);
  mx::Shape shape = a->shape();
  shape[0] = 1;
  return {{GetIntIndex(index, a->shape(0))},
          mx::broadcast_to(mx::reshape(std::move(update), std::move(up_shape)),
                           std::move(shape)),
          {0}};
}

// The implementation comes from mlx_scatter_args_array.
ScatterResult ScatterArgsArray(const mx::array* a,
                               mx::array indices,
                               mx::array update) {
  mx::Shape up_shape = GetUpShape(update);
  mx::array up = mx::reshape(std::move(update), std::move(up_shape));

  up_shape = indices.shape();
  up_shape.insert(up_shape.end(), a->shape().begin() + 1, a->shape().end());
  up = mx::broadcast_to(std::move(up), up_shape);
  up_shape.insert(up_shape.begin() + indices.ndim(), 1);
  up = mx::reshape(std::move(up), std::move(up_shape));

  return {{std::move(indices)}, std::move(up), {0}};
}


// The implementation comes from mlx_scatter_args_slice.
ScatterResult ScatterArgsSlice(const mx::array* a,
                               const Slice& slice,
                               mx::array update) {
  if (IsSliceNone(slice)) {
    mx::Shape up_shape = GetUpShape(update);
    return {{},
            mx::broadcast_to(mx::reshape(std::move(update),
                                         std::move(up_shape)),
                             a->shape()),
            {}};
  }

  mx::ShapeElem start = 0;
  mx::ShapeElem stop = a->shape(0);
  mx::ShapeElem step = 1;
  ReadSlice(slice, stop, &start, &stop, &step);

  if (step == 1) {
    mx::Shape up_shape = GetUpShape(update);
    mx::array up = mx::reshape(std::move(update), std::move(up_shape));

    mx::Shape up_shape_broadcast = {1, stop - start};
    up_shape_broadcast.insert(up_shape_broadcast.end(),
                              a->shape().begin() + 1, a->shape().end());
    return {{mx::array({start}, {1}, mx::uint32)},
            mx::broadcast_to(std::move(up), std::move(up_shape_broadcast)),
            {0}};
  }

  return ScatterArgsArray(a, mx::arange(start, stop, step, mx::uint32),
                          std::move(update));
}

// The implementation comes from mlx_scatter_args_nd.
ScatterResult ScatterArgsNDimentional(const mx::array* a,
                                      std::vector<ArrayIndex> entries,
                                      mx::array update) {
  auto [non_none_indices, indices] = ExpandEllipsis(a->shape(),
                                                    std::move(entries));
  if (non_none_indices > a->ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << a->ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  mx::Shape up_shape = GetUpShape(update);
  mx::array up = mx::reshape(std::move(update), std::move(up_shape));

  if (non_none_indices == 0) {
    return {{}, mx::broadcast_to(std::move(up), a->shape()), {}};
  }

  size_t max_dim = 0;
  bool arrays_first = false;
  size_t num_none = 0;
  size_t num_slices = 0;
  size_t num_arrays = 0;
  int num_strided_slices = 0;
  size_t num_simple_slices_post = 0;
  {
    bool have_array = false;
    bool have_non_array = false;
    for (const ArrayIndex& index : indices) {
      if (std::holds_alternative<std::monostate>(index)) {
        have_non_array = have_array;
        num_none++;
      } else if (std::holds_alternative<Slice>(index)) {
        have_non_array = have_array;
        num_slices++;
        if (std::get<Slice>(index).step.value_or(1) != 1) {
          num_strided_slices++;
          num_simple_slices_post = 0;
        } else {
          num_simple_slices_post++;
        }
      } else if (std::holds_alternative<mx::array*>(index)) {
        have_array = true;
        if (have_array && have_non_array)
          arrays_first = true;
        max_dim = std::max(std::get<mx::array*>(index)->ndim(), max_dim);
        num_arrays++;
        num_simple_slices_post = 0;
      }
    }
  }

  size_t index_ndim = max_dim + num_none + num_slices - num_simple_slices_post;
  if (index_ndim == 0)
    index_ndim = 1;

  std::vector<mx::array> arr_indices;
  size_t slice_num = 0;
  size_t array_num = 0;

  mx::Shape update_shape(non_none_indices, 1);
  mx::Shape slice_shapes;

  size_t axis = 0;
  for (const ArrayIndex& index : indices) {
    if (std::holds_alternative<Slice>(index)) {
      int start, stop, step;
      int axis_size = a->shape(axis);
      ReadSlice(std::get<Slice>(index), axis_size, &start, &stop, &step);

      start = start < 0 ? start + axis_size : start;
      stop = stop < 0 ? stop + axis_size : stop;

      mx::Shape index_shape(index_ndim, 1);
      if (array_num >= num_arrays && num_strided_slices == 0 && step == 1) {
        slice_shapes.push_back(stop - start);
        arr_indices.push_back(
            mx::array({start}, std::move(index_shape), mx::uint32));
        update_shape[axis] = slice_shapes.back();
      } else {
        mx::array idx = mx::arange(start, stop, step, mx::uint32);
        size_t loc = slice_num + (arrays_first ? max_dim : 0);
        index_shape[loc] = idx.size();
        arr_indices.push_back(mx::reshape(std::move(idx),
                                          std::move(index_shape)));
        slice_num++;
        num_strided_slices--;
        update_shape[axis] = 1;
      }
      axis++;
    } else if (std::holds_alternative<int>(index)) {
      arr_indices.push_back(GetIntIndex(std::get<int>(index),
                                        a->shape(axis++)));
      update_shape[axis - 1] = 1;
    } else if (std::holds_alternative<std::monostate>(index)) {
      slice_num++;
    } else if (std::holds_alternative<mx::array*>(index)) {
      mx::array* idx = std::get<mx::array*>(index);
      mx::Shape index_shape(index_ndim, 1);
      size_t start = (arrays_first ? 0 : 1) * slice_num + max_dim - idx->ndim();
      for (size_t j = 0; j < idx->ndim(); j++) {
        index_shape[start + j] = idx->shape(j);
      }
      arr_indices.push_back(mx::reshape(*idx, std::move(index_shape)));
      if (!arrays_first && ++array_num == num_arrays) {
        slice_num += max_dim;
      }
      update_shape[axis] = 1;
      axis++;
    } else {
      throw std::invalid_argument(
          "Cannot index mlx array using the given type yet");
    }
  }

  arr_indices = mx::broadcast_arrays(std::move(arr_indices));

  mx::Shape up_shape_broadcast = arr_indices[0].shape();
  up_shape_broadcast.insert(up_shape_broadcast.end(),
                            slice_shapes.begin(), slice_shapes.end());
  up_shape_broadcast.insert(up_shape_broadcast.end(),
                            a->shape().begin() + non_none_indices,
                            a->shape().end());
  up = mx::broadcast_to(std::move(up), std::move(up_shape_broadcast));

  mx::Shape up_reshape = arr_indices[0].shape();
  up_reshape.insert(up_reshape.end(), update_shape.begin(), update_shape.end());
  up_reshape.insert(up_reshape.end(),
                    a->shape().begin() + non_none_indices, a->shape().end());
  up = mx::reshape(std::move(up), std::move(up_reshape));

  mx::Shape axes(arr_indices.size(), 0);
  std::iota(axes.begin(), axes.end(), 0);
  return {std::move(arr_indices), std::move(up), std::move(axes)};
}

// The implementation comes from mlx_compute_scatter_args.
ScatterResult ComputeScatterArgs(const mx::array* a,
                                 std::variant<ArrayIndex, ArrayIndices> indices,
                                 ScalarOrArray vals) {
  mx::array value = ToArray(std::move(vals), a->dtype());
  if (std::holds_alternative<ArrayIndices>(indices)) {
    return ScatterArgsNDimentional(
        a, std::move(std::get<ArrayIndices>(indices)), std::move(value));
  }
  ArrayIndex index = std::move(std::get<ArrayIndex>(indices));
  if (std::holds_alternative<std::monostate>(index)) {
    return {{}, mx::broadcast_to(std::move(value), a->shape()), {}};
  }
  if (a->ndim() == 0) {
    throw std::invalid_argument("too many indices for array: "
                                "array is 0-dimensional");
  }
  if (std::holds_alternative<int>(index)) {
    return ScatterArgsInt(a, std::get<int>(index), std::move(value));
  }
  if (std::holds_alternative<mx::array*>(index)) {
    return ScatterArgsArray(a, std::move(*std::get<mx::array*>(index)),
                            std::move(value));
  }
  if (std::holds_alternative<Slice>(index)) {
    return ScatterArgsSlice(a, std::get<Slice>(index), std::move(value));
  }
  throw std::invalid_argument("Cannot index mlx array using the given type.");
}

// The implementation comes from mlx_slice_update.
std::pair<bool, mx::array> SliceUpdate(
    mx::array* a,
    std::variant<ArrayIndex, ArrayIndices> obj,
    ScalarOrArray vals) {
  bool is_slice = std::holds_alternative<ArrayIndex>(obj) &&
                  std::holds_alternative<Slice>(std::get<ArrayIndex>(obj));
  bool is_int = std::holds_alternative<ArrayIndex>(obj) &&
                std::holds_alternative<int>(std::get<ArrayIndex>(obj));
  // Can't route to slice update if not slice, tuple or int.
  if (a->ndim() == 0 ||
      (!is_slice && !is_int && !std::holds_alternative<ArrayIndices>(obj))) {
    return std::make_pair(false, *a);
  }
  if (std::holds_alternative<ArrayIndices>(obj)) {
    // Can't route to slice update if any arrays are present.
    for (const ArrayIndex& index : std::get<ArrayIndices>(obj)) {
      if (std::holds_alternative<mx::array*>(index))
        return std::make_pair(false, *a);
    }
  }

  // Should be able to route to slice update.

  // Pre process tuple.
  mx::array up = ToArray(std::move(vals), a->dtype());

  // Remove extra leading singletons dimensions from the update.
  int s = 0;
  while (s < up.ndim() && up.shape(s) == 1 && (up.ndim() - s) > a->ndim()) {
    s++;
  }
  mx::Shape up_shape(up.shape().begin() + s, up.shape().end());
  up = mx::reshape(std::move(up), up_shape.empty() ? mx::Shape{1}
                                                   : std::move(up_shape));

  // Build slice update params.
  mx::Shape starts(a->ndim(), 0);
  mx::Shape stops(a->shape());
  mx::Shape steps(a->ndim(), 1);
  if (is_int) {
    if (a->ndim() < 1) {
      std::ostringstream msg;
      msg << "Too many indices for array with " << a->ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    int idx = std::get<int>(std::get<ArrayIndex>(obj));
    idx = idx < 0 ? idx + stops[0] : idx;
    starts[0] = idx;
    stops[0] = idx + 1;
    return {true, mx::slice_update(*a, std::move(up), std::move(starts),
                                   std::move(stops), std::move(steps))};
  }

  // If it's just a simple slice, just do a slice update and return.
  if (is_slice) {
    ReadSlice(std::get<Slice>(std::get<ArrayIndex>(obj)), a->shape(0),
              &starts[0], &stops[0], &steps[0]);
    // Do slice update.
    return {true,
            mx::slice_update(*a, std::move(up), std::move(starts),
                             std::move(stops), std::move(steps))};
  }

  // It must be a tuple.
  ArrayIndices entries = std::move(std::get<ArrayIndices>(obj));

  // Expand ellipses into a series of ':' slices.
  auto [non_none_indices, indices] = ExpandEllipsis(a->shape(),
                                                    std::move(entries));
  // Dimension check.
  if (non_none_indices > a->ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << a->ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  // If no non-None indices return the broadcasted update.
  if (non_none_indices == 0) {
    return std::make_pair(true, mx::broadcast_to(std::move(up), a->shape()));
  }

  int unspecified = a->ndim() - non_none_indices;
  std::vector<int> squeeze_dims;
  std::vector<int> expand_dims;
  for (int i = indices.size() - 1,
           axis = non_none_indices - 1,
           up_axis = up.ndim() - unspecified - 1;
       i >= 0;
       --i) {
    const ArrayIndex& index = indices[i];
    if (std::holds_alternative<Slice>(index)) {
      ReadSlice(std::get<Slice>(index), a->shape(axis),
                &starts[axis], &stops[axis], &steps[axis]);
      axis--;
      up_axis--;
    } else if (std::holds_alternative<int>(index)) {
      int start = std::get<int>(index);
      if (start < 0)
        start += a->shape(i);
      starts[axis] = start;
      stops[axis] = start + 1;
      if (up_axis >= 0) {
        expand_dims.push_back(i - indices.size() - unspecified);
      }
      axis--;
    } else if (std::holds_alternative<std::monostate>(index)) {
      if (up_axis-- >= 0) {
        squeeze_dims.push_back(i - indices.size() - unspecified);
      }
    }
  }

  up = mx::squeeze(
      mx::expand_dims(up, std::move(expand_dims)), std::move(squeeze_dims));
  return {true,
          mx::slice_update(*a, std::move(up), std::move(starts),
                           std::move(stops), std::move(steps))};
}

}  // namespace

ArrayAt::ArrayAt(mx::array x, std::variant<ArrayIndex, ArrayIndices> indices)
    : x_(std::move(x)), indices_(std::move(indices)) {}

mx::array ArrayAt::Add(ScalarOrArray value) {
  auto [indices, updates, axes] = ComputeScatterArgs(&x_, indices_,
                                                     std::move(value));
  if (indices.size() > 0) {
    return mx::scatter_add(x_, std::move(indices), std::move(updates),
                           std::move(axes));
  } else {
    return mx::add(x_, std::move(updates));
  }
}

mx::array ArrayAt::Subtract(ScalarOrArray value) {
  auto [indices, updates, axes] = ComputeScatterArgs(&x_, indices_,
                                                     std::move(value));
  if (indices.size() > 0) {
    return mx::scatter_add(x_, std::move(indices),
                           mx::negative(std::move(updates)), std::move(axes));
  } else {
    return mx::subtract(x_, std::move(updates));
  }
}

mx::array ArrayAt::Multiply(ScalarOrArray value) {
  auto [indices, updates, axes] = ComputeScatterArgs(&x_, indices_,
                                                     std::move(value));
  if (indices.size() > 0) {
    return mx::scatter_prod(x_, std::move(indices), std::move(updates),
                            std::move(axes));
  } else {
    return mx::multiply(x_, std::move(updates));
  }
}

mx::array ArrayAt::Divide(ScalarOrArray value) {
  auto [indices, updates, axes] = ComputeScatterArgs(&x_, indices_,
                                                     std::move(value));
  if (indices.size() > 0) {
    return mx::scatter_prod(x_, std::move(indices),
                            mx::reciprocal(std::move(updates)),
                            std::move(axes));
  } else {
    return mx::divide(x_, std::move(updates));
  }
}

mx::array ArrayAt::Maximum(ScalarOrArray value) {
  auto [indices, updates, axes] = ComputeScatterArgs(&x_, indices_,
                                                     std::move(value));
  if (indices.size() > 0) {
    return mx::scatter_max(x_, std::move(indices), std::move(updates),
                           std::move(axes));
  } else {
    return mx::maximum(x_, std::move(updates));
  }
}

mx::array ArrayAt::Minimum(ScalarOrArray value) {
  auto [indices, updates, axes] = ComputeScatterArgs(&x_, indices_,
                                                     std::move(value));
  if (indices.size() > 0) {
    return mx::scatter_min(x_, std::move(indices), std::move(updates),
                           std::move(axes));
  } else {
    return mx::minimum(x_, std::move(updates));
  }
}

mx::array Index(const mx::array* a, ki::Arguments* args) {
  if (args->RemainingsLength() == 1) {
    auto index = args->GetNext<ArrayIndex>();
    if (!index) {
      args->ThrowError("Index");
      return *a;
    }
    return IndexOne(a, std::move(*index));
  }
  ArrayIndices indices;
  if (!ReadArgs(args, &indices)) {
    args->ThrowError("Index");
    return *a;
  }
  return IndexNDimensional(a, std::move(indices));
}

void IndexPut(mx::array* a,
              std::variant<ArrayIndex, ArrayIndices> obj,
              ScalarOrArray vals) {
  auto [success, out] = SliceUpdate(a, obj, vals);
  if (success) {
    a->overwrite_descriptor(std::move(out));
    return;
  }

  auto [indices, updates, axes] = ComputeScatterArgs(a, std::move(obj),
                                                     std::move(vals));
  if (indices.size() > 0) {
    a->overwrite_descriptor(mx::scatter(*a, std::move(indices),
                                        std::move(updates), std::move(axes)));
  } else {
    a->overwrite_descriptor(std::move(updates));
  }
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
