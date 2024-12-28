#include "src/array.h"
#include "src/ops.h"

// A template converter for binary ops.
inline
std::function<mx::array(ScalarOrArray a,
                        ScalarOrArray b,
                        mx::StreamOrDevice s)>
BinOpWrapper(mx::array(*func)(const mx::array&,
                              const mx::array&,
                              mx::StreamOrDevice)) {
  return [func](ScalarOrArray a, ScalarOrArray b, mx::StreamOrDevice s) {
    auto dtype = [&a, &b]() -> std::optional<mx::Dtype> {
      if (std::holds_alternative<mx::array>(a))
        return std::get<mx::array>(a).dtype();
      else if (std::holds_alternative<mx::array>(b))
        return std::get<mx::array>(b).dtype();
      else
        return std::nullopt;
    }();
    return func(ToArray(std::move(a), dtype), ToArray(std::move(b), dtype), s);
  };
}

// A template converter for ops that accept |k| and |axis|.
inline
std::function<mx::array(const mx::array& a,
                        int k,
                        std::optional<int> axis,
                        mx::StreamOrDevice s)>
KthOpWrapper(mx::array(*func1)(const mx::array&,
                               int,
                               int,
                               mx::StreamOrDevice),
             mx::array(*func2)(const mx::array&,
                               int,
                               mx::StreamOrDevice)) {
  return [func1, func2](const mx::array& a,
                        int k,
                        std::optional<int> axis,
                        mx::StreamOrDevice s) {
    if (axis)
      return func1(a, k, *axis, s);
    else
      return func2(a, k, s);
  };
}

// A template converter for |atleast_nd| ops.
inline
std::function<napi_value(napi_env env,
                         std::variant<mx::array, std::vector<mx::array>> arrays,
                         mx::StreamOrDevice s)>
NdOpWrapper(mx::array(*func1)(const mx::array&,
                              mx::StreamOrDevice),
            std::vector<mx::array>(*func2)(const std::vector<mx::array>&,
                                           mx::StreamOrDevice)) {
  return [func1, func2](napi_env env,
                        std::variant<mx::array, std::vector<mx::array>> arrays,
                        mx::StreamOrDevice s) {
    if (auto a = std::get_if<mx::array>(&arrays); a) {
      return ki::ToNodeValue(env, func1(*a, s));
    } else {
      return ki::ToNodeValue(
          env,
          func2(std::move(std::get<std::vector<mx::array>>(arrays)), s));
    }
  };
}

namespace ops {

mx::array Flatten(const mx::array& a,
                  std::optional<int> start_axis,
                  std::optional<int> end_axis,
                  mx::StreamOrDevice s) {
  return mx::flatten(a, start_axis.value_or(0), end_axis.value_or(-1), s);
}

mx::array Reshape(const mx::array& a,
                  ki::Arguments* args) {
  mx::Shape shape;
  if (args->RemainingsLength() == 1) {
    auto ret = args->GetNext<mx::Shape>();
    if (!ret) {
      args->ThrowError("Shape");
      return a;
    }
    shape = std::move(*ret);
  } else {
    if (!ReadArgs(args, &shape)) {
      args->ThrowError("int");
      return a;
    }
  }
  return mx::reshape(a, std::move(shape));
}

mx::array Squeeze(const mx::array& a,
                  OptionalAxes axis,
                  mx::StreamOrDevice s) {
  if (std::holds_alternative<std::monostate>(axis)) {
    return squeeze(a, s);
  } else if (auto i = std::get_if<int>(&axis); i) {
    return squeeze(a, *i, s);
  } else {
    return squeeze(a, std::move(std::get<std::vector<int>>(axis)), s);
  }
}

mx::array ExpandDims(const mx::array& a,
                     std::variant<int, std::vector<int>> dims,
                     mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&dims); i) {
    return expand_dims(a, *i, s);
  } else {
    return expand_dims(a, std::move(std::get<std::vector<int>>(dims)), s);
  }
}

mx::array ArrayEqual(const mx::array& a,
                     const mx::array& b,
                     std::optional<bool> equal_nan,
                     mx::StreamOrDevice s) {
  return mx::array_equal(a, b, equal_nan.value_or(false), s);
}

mx::array ARange(float first, ki::Arguments* args) {
  // The first arg is |stop| if arange is called with only 1 number.
  bool first_arg_is_start = false;
  if (args->Length() > 0)
    first_arg_is_start = ki::IsType(args->Env(), (*args)[1], napi_number);
  // Manual parsing args.
  float start = first_arg_is_start ? first : 0;
  float stop = first_arg_is_start ? args->GetNext<float>().value() : first;
  float step = args->TryGetNext<float>().value_or(1);
  auto dtype = args->TryGetNext<mx::Dtype>().value_or(mx::float32);
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  return mx::arange(start, stop, step, dtype, s);
}

mx::array Linspace(float start,
                   float stop,
                   std::optional<int> num,
                   std::optional<mx::Dtype> dtype,
                   mx::StreamOrDevice s) {
  return mx::linspace(start, stop, num.value_or(50),
                      dtype.value_or(mx::float32), s);
}

mx::array Take(const mx::array& a,
               const std::variant<int, mx::array>& indices,
               ki::Arguments* args) {
  auto axis = args->TryGetNext<int>();
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  if (auto i = std::get_if<int>(&indices); i) {
    return axis ? mx::take(a, *i, *axis, s) : mx::take(a, *i, s);
  } else {
    const mx::array& indicesArr = std::get<mx::array>(indices);
    return axis ? mx::take(a, indicesArr, *axis, s)
                : mx::take(a, indicesArr, s);
  }
}

mx::array TakeAlongAxis(const mx::array& a,
                        const mx::array& indices,
                        ki::Arguments* args) {
  auto axis = args->TryGetNext<int>();
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  if (axis)
    return mx::take_along_axis(a, indices, *axis, s);
  else
    return mx::take_along_axis(mx::reshape(a, {-1}, s), indices, 0, s);
}

mx::array PutAlongAxis(const mx::array& a,
                       const mx::array& indices,
                       const mx::array& values,
                       ki::Arguments* args) {
  auto axis = args->TryGetNext<int>();
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  if (axis)
    return mx::put_along_axis(a, indices, values, *axis, s);
  else
    return mx::reshape(mx::put_along_axis(reshape(a, {-1}, s),
                                          indices, values, 0, s),
                       a.shape(), s);
}

mx::array Full(std::variant<int, mx::Shape> shape,
               ScalarOrArray vals,
               std::optional<mx::Dtype> dtype,
               mx::StreamOrDevice s) {
  return mx::full(PutIntoVector(std::move(shape)),
                  ToArray(std::move(vals), std::move(dtype)),
                  s);
}

mx::array Zeros(std::variant<int, mx::Shape> shape,
                std::optional<mx::Dtype> dtype,
                mx::StreamOrDevice s) {
  return mx::zeros(PutIntoVector(std::move(shape)), dtype.value_or(mx::float32), s);
}

mx::array Ones(std::variant<int, mx::Shape> shape,
               std::optional<mx::Dtype> dtype,
               mx::StreamOrDevice s) {
  return mx::ones(PutIntoVector(std::move(shape)), dtype.value_or(mx::float32), s);
}

mx::array Eye(int n,
              std::optional<int> m,
              std::optional<int> k,
              std::optional<mx::Dtype> dtype,
              mx::StreamOrDevice s) {
  return mx::eye(n, m.value_or(n), k.value_or(0), dtype.value_or(mx::float32),
                 s);
}

mx::array Identity(int n,
                   std::optional<mx::Dtype> dtype,
                   mx::StreamOrDevice s) {
  return mx::identity(n, dtype.value_or(mx::float32), s);
}

mx::array AllClose(const mx::array& a,
                   const mx::array& b,
                   std::optional<double> rtol,
                   std::optional<double> atol,
                   std::optional<bool> equal_nan,
                   mx::StreamOrDevice s) {
  return mx::allclose(a, b, rtol.value_or(1e-5), atol.value_or(1e-8),
                      equal_nan.value_or(false), s);
}

mx::array IsClose(const mx::array& a,
                  const mx::array& b,
                  std::optional<double> rtol,
                  std::optional<double> atol,
                  std::optional<bool> equal_nan,
                  mx::StreamOrDevice s) {
  return mx::isclose(a, b, rtol.value_or(1e-5), atol.value_or(1e-8),
                     equal_nan.value_or(false), s);
}

mx::array Tri(int n,
              std::optional<int> m,
              std::optional<int> k,
              std::optional<mx::Dtype> dtype,
              mx::StreamOrDevice s) {
  return mx::tri(n, m.value_or(n), k.value_or(0), dtype.value_or(mx::float32),
                 s);
}

mx::array Transpose(const mx::array& a,
                    ki::Arguments* args) {
  std::vector<int> axes;
  if (args->RemainingsLength() == 1) {
    // Both a.transpose(1) and a.transpose([1]) should work.
    if (auto v = args->GetNext<std::vector<int>>(); v) {
      axes = std::move(*v);
    } else if (auto i = args->GetNext<int>(); v) {
      axes = {*i};
    } else {
      args->ThrowError("Axes");
      return a;
    }
  } else {
    // Make a.transpose(1, 2, 3) work.
    if (!ReadArgs(args, &axes)) {
      args->ThrowError("int");
      return a;
    }
  }
  if (axes.empty())
    return mx::transpose(a);
  else
    return mx::transpose(a, std::move(axes));
}

mx::array Std(const mx::array& a,
              OptionalAxes axis,
              std::optional<bool> keepdims,
              std::optional<int> ddof,
              mx::StreamOrDevice s) {
  return mx::std(a, GetReduceAxes(std::move(axis), a.ndim()),
                 keepdims.value_or(false), ddof.value_or(0), s);
}

mx::array Var(const mx::array& a,
              OptionalAxes axis,
              std::optional<bool> keepdims,
              std::optional<int> ddof,
              mx::StreamOrDevice s) {
  return mx::var(a, GetReduceAxes(std::move(axis), a.ndim()),
                 keepdims.value_or(false), ddof.value_or(0), s);
}

std::vector<mx::array> Split(const mx::array& a,
                             std::variant<int, std::vector<int>> indices,
                             std::optional<int> axis,
                             mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&indices); i) {
    return mx::split(a, *i, axis.value_or(0), s);
  } else {
    return mx::split(a, std::move(std::get<std::vector<int>>(indices)),
                     axis.value_or(0), s);
  }
}

mx::array ArgMin(const mx::array& a, ki::Arguments* args) {
  auto axis = args->TryGetNext<int>();
  auto keepdims = args->TryGetNext<bool>().value_or(false);
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  if (axis)
    return mx::argmin(a, *axis, keepdims, s);
  else
    return mx::argmin(a, keepdims, s);
}

mx::array ArgMax(const mx::array& a, ki::Arguments* args) {
  auto axis = args->TryGetNext<int>();
  auto keepdims = args->TryGetNext<bool>().value_or(false);
  auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
  if (axis)
    return mx::argmax(a, *axis, keepdims, s);
  else
    return mx::argmax(a, keepdims, s);
}

mx::array Sort(const mx::array& a,
               std::optional<int> axis,
               mx::StreamOrDevice s) {
  if (axis)
    return mx::sort(a, *axis, s);
  else
    return mx::sort(a, s);
}

mx::array ArgSort(const mx::array& a,
                  std::optional<int> axis,
                  mx::StreamOrDevice s) {
  if (axis)
    return mx::argsort(a, *axis, s);
  else
    return mx::argsort(a, s);
}

mx::array Softmax(const mx::array& a,
                  OptionalAxes axis,
                  std::optional<bool> precise,
                  mx::StreamOrDevice s) {
  return mx::softmax(a, GetReduceAxes(std::move(axis), a.ndim()),
                     precise.value_or(false), s);
}

mx::array Concatenate(std::vector<mx::array> arrays,
                      std::optional<int> axis,
                      mx::StreamOrDevice s) {
  if (axis)
    return mx::concatenate(std::move(arrays), *axis, s);
  else
    return mx::concatenate(std::move(arrays), s);
}

mx::array Stack(std::vector<mx::array> arrays,
                std::optional<int> axis,
                mx::StreamOrDevice s) {
  if (axis)
    return mx::stack(std::move(arrays), *axis, s);
  else
    return mx::stack(std::move(arrays), s);
}

std::vector<mx::array> Meshgrid(ki::Arguments* args) {
  std::vector<mx::array> arrays;
  ReadArgs(args, &arrays);
  return mx::meshgrid(
      std::move(arrays),
      args->TryGetNext<bool>().value_or(false),
      args->TryGetNext<std::string>().value_or("xy"),
      args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate()));
}

mx::array Pad(const mx::array& a,
              std::variant<int,
                           std::tuple<int>,
                           std::pair<int, int>,
                           std::vector<std::pair<int, int>>> width,
              std::optional<mx::array> optional_constant,
              std::optional<std::string> optional_mode,
              mx::StreamOrDevice s) {
  mx::array constant = std::move(optional_constant.value_or(mx::array(0)));
  std::string mode = optional_mode.value_or("constant");
  if (auto i = std::get_if<int>(&width); i)
    return mx::pad(a, *i, constant, mode, s);
  if (auto ti = std::get_if<std::tuple<int>>(&width); ti)
    return mx::pad(a, std::get<0>(*ti), constant, mode, s);
  if (auto tii = std::get_if<std::pair<int, int>>(&width); tii)
    return mx::pad(a, *tii, constant, mode, s);
  auto v = std::get<std::vector<std::pair<int, int>>>(width);
  if (v.size() == 1)
    return mx::pad(a, v[0], constant, mode, s);
  else
    return mx::pad(a, v, constant, mode, s);
}

mx::array AsStrided(const mx::array& a,
                    std::optional<mx::Shape> shape,
                    std::optional<mx::Strides> strides,
                    std::optional<size_t> offset,
                    mx::StreamOrDevice s) {
  mx::Shape a_shape = shape.value_or(a.shape());
  mx::Strides a_strides;
  if (strides) {
    a_strides = std::move(*strides);
  } else {
    a_strides = mx::Strides(a_shape.size(), 1);
    for (int i = a_shape.size() - 1; i > 0; i--)
      a_strides[i - 1] = a_shape[i] * a_strides[i];
  }
  return as_strided(a, a_shape, a_strides, offset.value_or(0), s);
}

mx::array Convolve(const mx::array& a,
                   const mx::array& v,
                   std::optional<std::string> mode,
                   mx::StreamOrDevice s) {
  if (a.ndim() != 1 || v.ndim() != 1)
    throw std::invalid_argument("[convolve] Inputs must be 1D.");
  if (a.size() == 0 || v.size() == 0)
    throw std::invalid_argument("[convolve] Inputs cannot be empty.");

  mx::array in = a.size() < v.size() ? v : a;
  mx::array wt = a.size() < v.size() ? a : v;
  wt = mx::slice(wt, {wt.shape(0) - 1}, {-wt.shape(0) - 1}, {-1}, s);

  in = mx::reshape(std::move(in), {1, -1, 1}, s);
  wt = mx::reshape(std::move(wt), {1, -1, 1}, s);

  int padding = 0;

  std::string m = mode.value_or("full");
  if (m == "full") {
    padding = wt.size() - 1;
  } else if (m == "valid") {
    padding = 0;
  } else if (m == "same") {
    if (wt.size() % 2) {
      padding = wt.size() / 2;
    } else {
      int pad_l = wt.size() / 2;
      int pad_r = std::max(0, pad_l - 1);
      in = mx::pad(in, {{0, 0}, {pad_l, pad_r}, {0, 0}}, mx::array(0), "constant", s);
    }
  } else {
    throw std::invalid_argument("[convolve] Invalid mode.");
  }

  mx::array out = mx::conv1d(in, wt, /*stride = */ 1, /*padding = */ padding,
                             /*dilation = */ 1, /*groups = */ 1, s);
  return reshape(out, {-1}, s);
}

mx::array Conv1d(
    const mx::array& input,
    const mx::array& weight,
    std::optional<int> stride,
    std::optional<int> padding,
    std::optional<int> dilation,
    std::optional<int> groups,
    mx::StreamOrDevice s) {
  return mx::conv1d(input, weight, stride.value_or(1), padding.value_or(0),
                    dilation.value_or(1), groups.value_or(1), s);
}

mx::array Conv2d(
    const mx::array& input,
    const mx::array& weight,
    std::variant<std::monostate, int, std::pair<int, int>> stride,
    std::variant<std::monostate, int, std::pair<int, int>> padding,
    std::variant<std::monostate, int, std::pair<int, int>> dilation,
    std::optional<int> groups,
    mx::StreamOrDevice s) {
  std::pair<int, int> stride_pair(1, 1);
  if (auto i = std::get_if<int>(&stride); i)
    stride_pair = std::pair<int, int>(*i, *i);
  else if (auto p = std::get_if<std::pair<int, int>>(&stride); p)
    stride_pair = std::move(*p);

  std::pair<int, int> padding_pair(0, 0);
  if (auto i = std::get_if<int>(&padding); i)
    padding_pair = std::pair<int, int>(*i, *i);
  else if (auto p = std::get_if<std::pair<int, int>>(&padding); p)
    padding_pair = std::move(*p);

  std::pair<int, int> dilation_pair(1, 1);
  if (auto i = std::get_if<int>(&dilation); i)
    dilation_pair = std::pair<int, int>(*i, *i);
  else if (auto p = std::get_if<std::pair<int, int>>(&dilation); p)
    dilation_pair = std::move(*p);

  return mx::conv2d(input, weight, stride_pair, padding_pair, dilation_pair,
                    groups.value_or(1), s);
}

mx::array Conv3d(
    const mx::array& input,
    const mx::array& weight,
    std::variant<std::monostate, int, std::tuple<int, int, int>> stride,
    std::variant<std::monostate, int, std::tuple<int, int, int>> padding,
    std::variant<std::monostate, int, std::tuple<int, int, int>> dilation,
    std::optional<int> groups,
    mx::StreamOrDevice s) {
  std::tuple<int, int, int> stride_tuple = {1, 1, 1};
  if (auto i = std::get_if<int>(&stride); i)
    stride_tuple = {*i, *i, *i};
  else if (auto p = std::get_if<std::tuple<int, int, int>>(&stride); p)
    stride_tuple = std::move(*p);

  std::tuple<int, int, int> padding_tuple = {0, 0, 0};
  if (auto i = std::get_if<int>(&padding); i)
    padding_tuple = {*i, *i, *i};
  else if (auto p = std::get_if<std::tuple<int, int, int>>(&padding); p)
    padding_tuple = std::move(*p);

  std::tuple<int, int, int> dilation_tuple = {1, 1, 1};
  if (auto i = std::get_if<int>(&dilation); i)
    dilation_tuple = {*i, *i, *i};
  else if (auto p = std::get_if<std::tuple<int, int, int>>(&dilation); p)
    dilation_tuple = std::move(*p);

  return mx::conv3d(input, weight, stride_tuple, padding_tuple, dilation_tuple,
                    groups.value_or(1), s);
}

mx::array ConvTranspose1d(
    const mx::array& input,
    const mx::array& weight,
    std::optional<int> stride,
    std::optional<int> padding,
    std::optional<int> dilation,
    std::optional<int> groups,
    mx::StreamOrDevice s) {
  return mx::conv_transpose1d(input, weight, stride.value_or(1),
                              padding.value_or(0), dilation.value_or(1),
                              groups.value_or(1), s);
}

mx::array ConvTranspose2d(
    const mx::array& input,
    const mx::array& weight,
    std::variant<std::monostate, int, std::pair<int, int>> stride,
    std::variant<std::monostate, int, std::pair<int, int>> padding,
    std::variant<std::monostate, int, std::pair<int, int>> dilation,
    std::optional<int> groups,
    mx::StreamOrDevice s) {
  std::pair<int, int> stride_pair(1, 1);
  if (auto i = std::get_if<int>(&stride); i)
    stride_pair = std::pair<int, int>(*i, *i);
  else if (auto p = std::get_if<std::pair<int, int>>(&stride); p)
    stride_pair = std::move(*p);

  std::pair<int, int> padding_pair(0, 0);
  if (auto i = std::get_if<int>(&padding); i)
    padding_pair = std::pair<int, int>(*i, *i);
  else if (auto p = std::get_if<std::pair<int, int>>(&padding); p)
    padding_pair = std::move(*p);

  std::pair<int, int> dilation_pair(1, 1);
  if (auto i = std::get_if<int>(&dilation); i)
    dilation_pair = std::pair<int, int>(*i, *i);
  else if (auto p = std::get_if<std::pair<int, int>>(&dilation); p)
    dilation_pair = std::move(*p);

  return mx::conv_transpose2d(input, weight, stride_pair, padding_pair,
                              dilation_pair, groups.value_or(1), s);
}

mx::array ConvTranspose3d(
    const mx::array& input,
    const mx::array& weight,
    std::variant<std::monostate, int, std::tuple<int, int, int>> stride,
    std::variant<std::monostate, int, std::tuple<int, int, int>> padding,
    std::variant<std::monostate, int, std::tuple<int, int, int>> dilation,
    std::optional<int> groups,
    mx::StreamOrDevice s) {
  std::tuple<int, int, int> stride_tuple = {1, 1, 1};
  if (auto i = std::get_if<int>(&stride); i)
    stride_tuple = {*i, *i, *i};
  else if (auto p = std::get_if<std::tuple<int, int, int>>(&stride); p)
    stride_tuple = std::move(*p);

  std::tuple<int, int, int> padding_tuple = {0, 0, 0};
  if (auto i = std::get_if<int>(&padding); i)
    padding_tuple = {*i, *i, *i};
  else if (auto p = std::get_if<std::tuple<int, int, int>>(&padding); p)
    padding_tuple = std::move(*p);

  std::tuple<int, int, int> dilation_tuple = {1, 1, 1};
  if (auto i = std::get_if<int>(&dilation); i)
    dilation_tuple = {*i, *i, *i};
  else if (auto p = std::get_if<std::tuple<int, int, int>>(&dilation); p)
    dilation_tuple = std::move(*p);

  return mx::conv_transpose3d(input, weight, stride_tuple, padding_tuple,
                              dilation_tuple, groups.value_or(1), s);
}

mx::array ConvGeneral(
    const mx::array& input,
    const mx::array& weight,
    std::optional<std::variant<int, std::vector<int>>> stride,
    std::variant<std::monostate,
                 int,
                 std::vector<int>,
                 std::pair<std::vector<int>, std::vector<int>>> padding,
    std::optional<std::variant<int, std::vector<int>>> kernel_dilation,
    std::optional<std::variant<int, std::vector<int>>> input_dilation,
    std::optional<int> groups,
    std::optional<bool> flip,
    mx::StreamOrDevice s) {
  std::vector<int> padding_lo_vec = {0};
  std::vector<int> padding_hi_vec = {0};
  if (auto i = std::get_if<int>(&padding); i) {
    padding_lo_vec = {*i};
    padding_hi_vec = {*i};
  } else if (auto v = std::get_if<std::vector<int>>(&padding); v) {
    padding_lo_vec = std::move(*v);
    padding_hi_vec = padding_lo_vec;
  } else if (auto p = std::get_if<std::pair<std::vector<int>,
                                            std::vector<int>>>(&padding);
             p) {
    padding_lo_vec = std::move(p->first);
    padding_hi_vec = std::move(p->second);
  }

  return mx::conv_general(std::move(input),
                          std::move(weight),
                          PutIntoVector(std::move(stride.value_or(1))),
                          std::move(padding_lo_vec),
                          std::move(padding_hi_vec),
                          PutIntoVector(std::move(kernel_dilation.value_or(1))),
                          PutIntoVector(std::move(input_dilation.value_or(1))),
                          groups.value_or(1),
                          flip.value_or(false),
                          s);
}

mx::array Round(const mx::array& a,
                std::optional<int> decimals,
                mx::StreamOrDevice s) {
  return mx::round(a, decimals.value_or(0), s);
}

mx::array Tensordot(const mx::array& a,
                    const mx::array& b,
                    std::variant<std::monostate,
                                 int,
                                 std::vector<std::vector<int>>> axes,
                    mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&axes); i)
    return mx::tensordot(a, b, *i, s);
  if (auto v = std::get_if<std::vector<std::vector<int>>>(&axes); v) {
    if (v->size() != 2) {
      throw std::invalid_argument(
          "[tensordot] axes must be a list of two lists.");
    }
    return mx::tensordot(a, b, (*v)[0], (*v)[1], s);
  }
  // The |axes| arg default to 2.
  return mx::tensordot(a, b, 2, s);
}

mx::array Tile(const mx::array& a,
               std::variant<int, std::vector<int>> reps,
               mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&reps); i)
    return mx::tile(a, {*i}, s);
  else
    return mx::tile(a, std::get<std::vector<int>>(reps), s);
}

mx::array AddMM(mx::array a,
                mx::array b,
                mx::array c,
                std::optional<float> alpha,
                std::optional<float> beta,
                mx::StreamOrDevice s) {
  return mx::addmm(std::move(a), std::move(b), std::move(c),
                   alpha.value_or(1), beta.value_or(1), s);
}

mx::array Diagonal(const mx::array& a,
                   std::optional<int> offset,
                   std::optional<int> axis1,
                   std::optional<int> axis2,
                   mx::StreamOrDevice s) {
  return mx::diagonal(a, offset.value_or(0), axis1.value_or(0),
                      axis2.value_or(1), s);
}

mx::array Diag(const mx::array& a,
               std::optional<int> k,
               mx::StreamOrDevice s) {
  return mx::diag(a, k.value_or(0), s);
}

mx::array Trace(const mx::array& a,
                int offset,
                int axis1,
                int axis2,
                std::optional<mx::Dtype> dtype,
                mx::StreamOrDevice s) {
  if (dtype)
    return mx::trace(a, offset, axis1, axis2, dtype.value(), s);
  else
    return mx::trace(a, offset, axis1, axis2, s);
}

bool IsSubDtype(std::variant<mx::Dtype, mx::Dtype::Category> dtype,
                std::variant<mx::Dtype, mx::Dtype::Category> category) {
  if (auto d = std::get_if<mx::Dtype>(&dtype); d) {
    if (auto c = std::get_if<mx::Dtype>(&category); c)
      return mx::issubdtype(*d, *c);
    else
      return mx::issubdtype(*d, std::get<mx::Dtype::Category>(category));
  } else {
    if (auto c = std::get_if<mx::Dtype>(&category); c)
      return mx::issubdtype(std::get<mx::Dtype::Category>(dtype), *c);
    else
      return mx::issubdtype(std::get<mx::Dtype::Category>(dtype),
                            std::get<mx::Dtype::Category>(category));
  }
}

mx::array Roll(const mx::array& a,
               OptionalAxes shift,
               OptionalAxes axis,
               mx::StreamOrDevice s) {
  return std::visit(
      [&](auto sh, auto ax) -> mx::array {
        using T = decltype(ax);
        using V = decltype(sh);

        if constexpr (std::is_same_v<V, std::monostate>) {
          throw std::invalid_argument(
              "[roll] Expected two arguments but only one was given.");
        } else {
          if constexpr (std::is_same_v<T, std::monostate>) {
            return mx::roll(a, sh, s);
          } else {
            return mx::roll(a, sh, ax, s);
          }
        }
      },
      shift,
      axis);
}

}  // namespace ops

void InitOps(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "reshape", &ops::Reshape,
          "flatten", &ops::Flatten,
          "squeeze", &ops::Squeeze,
          "expandDims", &ops::ExpandDims,
          "abs", &mx::abs,
          "sign", &mx::sign,
          "negative", &mx::negative,
          "add", BinOpWrapper(&mx::add),
          "subtract", BinOpWrapper(&mx::subtract),
          "multiply", BinOpWrapper(&mx::multiply),
          "divide", BinOpWrapper(&mx::divide),
          "divmod", &mx::divmod,
          "floorDivide", BinOpWrapper(&mx::floor_divide),
          "remainder", BinOpWrapper(&mx::remainder),
          "equal", BinOpWrapper(&mx::equal),
          "notEqual", BinOpWrapper(&mx::not_equal),
          "less", BinOpWrapper(&mx::less),
          "lessEqual", BinOpWrapper(&mx::less_equal),
          "greater", BinOpWrapper(&mx::greater),
          "greaterEqual", BinOpWrapper(&mx::greater_equal),
          "arrayEqual", &ops::ArrayEqual,
          "matmul", &mx::matmul,
          "square", &mx::square,
          "sqrt", &mx::sqrt,
          "rsqrt", &mx::rsqrt,
          "reciprocal", &mx::reciprocal,
          "logicalNot", &mx::logical_not,
          "logicalAnd", &mx::logical_and,
          "logicalOr", &mx::logical_or,
          "logaddexp", BinOpWrapper(&mx::logaddexp),
          "exp", &mx::exp,
          "expm1", &mx::expm1,
          "erf", &mx::erf,
          "erfinv", &mx::erfinv,
          "sin", &mx::sin,
          "cos", &mx::cos,
          "tan", &mx::tan,
          "arcsin", &mx::arcsin,
          "arccos", &mx::arccos,
          "arctan", &mx::arctan,
          "arctan2", &mx::arctan2,
          "sinh", &mx::sinh,
          "cosh", &mx::cosh,
          "tanh", &mx::tanh,
          "arcsinh", &mx::arcsinh,
          "arccosh", &mx::arccosh,
          "arctanh", &mx::arctanh,
          "degrees", &mx::degrees,
          "radians", &mx::radians,
          "log", &mx::log,
          "log2", &mx::log2,
          "log10", &mx::log10,
          "log1p", &mx::log1p,
          "stopGradient", &mx::stop_gradient,
          "sigmoid", &mx::sigmoid,
          "power", BinOpWrapper(&mx::power),
          "arange", &ops::ARange,
          "linspace", &ops::Linspace,
          "take", &ops::Take,
          "takeAlongAxis", &ops::TakeAlongAxis,
          "putAlongAxis", &ops::PutAlongAxis,
          "full", &ops::Full,
          "zeros", &ops::Zeros,
          "zerosLike", &mx::zeros_like,
          "ones", &ops::Ones,
          "onesLike", &mx::ones_like,
          "eye", &ops::Eye,
          "identity", &ops::Identity,
          "tri", &ops::Tri,
          "tril", &mx::tril,
          "allclose", &ops::AllClose,
          "isclose", &ops::IsClose,
          "all", DimOpWrapper(&mx::all),
          "any", DimOpWrapper(&mx::any),
          "minimum", BinOpWrapper(&mx::minimum),
          "maximum", BinOpWrapper(&mx::maximum),
          "floor", &mx::floor,
          "ceil", &mx::ceil,
          "isnan", &mx::isnan,
          "isinf", &mx::isinf,
          "isfinite", &mx::isfinite,
          "isposinf", &mx::isposinf,
          "isneginf", &mx::isneginf,
          "moveaxis", &mx::moveaxis,
          "swapaxes", &mx::swapaxes,
          "transpose", &ops::Transpose,
          "permuteDims", &ops::Transpose,
          "sum", DimOpWrapper(&mx::sum),
          "prod", DimOpWrapper(&mx::prod),
          "min", DimOpWrapper(&mx::min),
          "max", DimOpWrapper(&mx::max),
          "logsumexp", DimOpWrapper(&mx::logsumexp),
          "mean", DimOpWrapper(&mx::mean),
          "variance", &ops::Var,
          "std", &ops::Std,
          "split", &ops::Split,
          "argmin", &ops::ArgMin,
          "argmax", &ops::ArgMax,
          "sort", &ops::Sort,
          "argsort", &ops::ArgSort,
          "partition", KthOpWrapper(&mx::partition, &mx::partition),
          "argpartition", KthOpWrapper(&mx::argpartition, &mx::argpartition),
          "topk", KthOpWrapper(&mx::topk, &mx::topk),
          "broadcastTo", &mx::broadcast_to,
          "blockMaskedMM", &mx::block_masked_mm,
          "gatherMM", &mx::gather_mm,
          "gatherQMM", &mx::gather_qmm,
          "softmax", &ops::Softmax,
          "concatenate", &ops::Concatenate,
          "concat", &ops::Concatenate,
          "stack", &ops::Stack,
          "meshgrid", &ops::Meshgrid,
          "repeat", KthOpWrapper(&mx::repeat, &mx::repeat),
          "clip", &mx::clip,
          "pad", &ops::Pad,
          "asStrided", &ops::AsStrided,
          "cumsum", CumOpWrapper(&mx::cumsum),
          "cumprod", CumOpWrapper(&mx::cumprod),
          "cummax", CumOpWrapper(&mx::cummax),
          "cummin", CumOpWrapper(&mx::cummin),
          "conj", &mx::conjugate,
          "conjugate", &mx::conjugate,
          "convolve", &ops::Convolve,
          "conv1d", &ops::Conv1d,
          "conv2d", &ops::Conv2d,
          "conv3d", &ops::Conv3d,
          "convTranspose1d", &ops::ConvTranspose1d,
          "convTranspose2d", &ops::ConvTranspose2d,
          "convTranspose3d", &ops::ConvTranspose3d,
          "convGeneral", &ops::ConvGeneral,
          "where", &mx::where,
          "nanToNum", &mx::nan_to_num,
          "round", &ops::Round,
          "quantizedMatmul", &mx::quantized_matmul,
          "quantize", &mx::quantize,
          "dequantize", &mx::dequantize,
          "tensordot", &ops::Tensordot,
          "inner", &mx::inner,
          "outer", &mx::outer,
          "tile", &ops::Tile,
          "addmm", &ops::AddMM,
          "blockMaskedMM", &mx::block_masked_mm,
          "diagonal", &ops::Diagonal,
          "diag", &ops::Diag,
          "trace", &ops::Trace,
          "atleast1d", NdOpWrapper(&mx::atleast_1d, &mx::atleast_1d),
          "atleast2d", NdOpWrapper(&mx::atleast_1d, &mx::atleast_2d),
          "atleast3d", NdOpWrapper(&mx::atleast_1d, &mx::atleast_3d),
          "issubdtype", &ops::IsSubDtype,
          "bitwiseAnd", BinOpWrapper(&mx::bitwise_and),
          "bitwiseOr", BinOpWrapper(&mx::bitwise_or),
          "bitwiseXor", BinOpWrapper(&mx::bitwise_xor),
          "leftShift", BinOpWrapper(&mx::left_shift),
          "rightShift", BinOpWrapper(&mx::right_shift),
          "view", &mx::view,
          "hadamardTransform", &mx::hadamard_transform,
          "einsumPath", &mx::einsum_path,
          "einsum", &mx::einsum,
          "roll", &ops::Roll,
          "real", &mx::real,
          "imag", &mx::imag);
}
