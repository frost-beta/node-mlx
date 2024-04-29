#include "src/array.h"
#include "src/ops.h"
#include "src/trees.h"

// Needed for detail::compile.
#include "mlx/transforms_impl.h"

namespace {

// Unflatten the function call result.
inline napi_value UnflattenResults(napi_env env,
                                   const std::vector<mx::array>& results) {
  if (results.size() > 1)
    return ki::ToNodeValue(env, results);
  else
    return ki::ToNodeValue(env, results[0]);
}

// Execute JS function with primals.
std::vector<mx::array> ExecuteWithPrimals(
    napi_env env,
    napi_value js_func,
    const std::vector<mx::array>& primals) {
  // Convert primals to JS values.
  std::vector<napi_value> args;
  for (const mx::array& primal : primals)
    args.push_back(ki::ToNodeValue(env, primal));
  // Call the JS function with primals.
  napi_value result;
  if (napi_make_callback(env, nullptr, js_func, js_func,
                         args.size(), args.empty() ? nullptr : &args.front(),
                         &result) != napi_ok) {
    return {};
  }
  // Convert result to vector.
  if (auto a = ki::FromNodeTo<mx::array*>(env, result); a)
    return std::vector<mx::array>{*a.value()};
  if (auto v = ki::FromNodeTo<std::vector<mx::array>>(env, result); v)
    return std::move(*v);
  ki::ThrowError(env, "function does not return mx.array or Array of mx.array");
  return {};
}

// A template converter for ops that accept infinite |array|s.
inline
std::function<void(ki::Arguments* args)>
EvalOpWrapper(void(*func)(std::vector<mx::array>)) {
  return [func](ki::Arguments* args) {
    func(TreeFlatten(args));
  };
}

// A template converter for ops that accept |primals| and |tangents|.
inline
std::function<
    std::pair<std::vector<mx::array>, std::vector<mx::array>>(
        napi_env env,
        napi_value func,
        std::vector<mx::array> primals,
        std::vector<mx::array> tangents)>
JVPOpWrapper(
    std::pair<std::vector<mx::array>, std::vector<mx::array>> (*func)(
        const std::function<
            std::vector<mx::array>(const std::vector<mx::array>&)>&,
        const std::vector<mx::array>&,
        const std::vector<mx::array>&)) {
  return [func](napi_env env,
                napi_value js_func,
                std::vector<mx::array> primals,
                std::vector<mx::array> tangents) {
    auto vfunc = [env, js_func](const std::vector<mx::array>& primals) {
      return ExecuteWithPrimals(env, js_func, primals);
    };
    return func(vfunc, primals, tangents);
  };
}

}  // namespace

namespace transforms_ops {

std::function<std::pair<napi_value, napi_value>(ki::Arguments)>
ValueAndGradImpl(const char* error_tag,
                 bool scalar_func_only,
                 ki::Persistent js_func,
                 std::optional<std::variant<int, std::vector<int>>> arg) {
  // Sanitize |argnums|.
  std::vector<int> argnums = ToIntVector(std::move(arg.value_or(0)));
  if (argnums.size() > 0) {
    std::sort(argnums.begin(), argnums.end());
    if (argnums[0] < 0) {
      ki::ThrowError(js_func.Env(),
                     error_tag, " Can't compute the gradient of negative "
                     "argument index ", argnums[0]);
      return nullptr;
    }
  }
  return [error_tag,
          scalar_func_only,
          js_func = std::move(js_func),
          argnums = std::move(argnums)](ki::Arguments args)
      -> std::pair<napi_value, napi_value> {
    // Check |argnums| is within the arguments length.
    if (argnums.size() > 0 && argnums.back() >= args.Length()) {
      ki::ThrowError(js_func.Env(),
                     error_tag, " Can't compute the gradient of argument "
                     "index ", argnums.back(), " because the function is "
                     "called with only ", args.Length(), " arguments.");
      return {nullptr, nullptr};
    }
    // Collect the arrays located at |argnums|.
    std::vector<mx::array> arrays;
    std::vector<size_t> strides = {0};
    for (int i : argnums) {
      // Concatenate flatten results to |arrays|.
      std::vector<mx::array> flat = TreeFlatten(args.Env(), args[i]);
      std::move(flat.begin(), flat.end(), std::back_inserter(arrays));
      strides.push_back(flat.size());
    }
    // Calculate strides.
    std::partial_sum(strides.cbegin(), strides.cend(), strides.begin());
    // Generate indices for every element in |arrays|.
    std::vector<int> gradient_indices(arrays.size());
    std::iota(gradient_indices.begin(), gradient_indices.end(), 0);
    // The result of |js_func| execution.
    napi_value result = nullptr;
    // Call value_and_grad with the JS function.
    napi_env env = js_func.Env();
    auto value_and_grad_func = mx::value_and_grad(
        [error_tag, scalar_func_only,
         &js_func, &args, &argnums, &arrays, &strides, &result, &env](
            const std::vector<mx::array>& primals) -> std::vector<mx::array> {
          // Read the args into |js_args| vector, and replace the arrays in it
          // with the traced |primals|.
          std::vector<napi_value> js_args;
          size_t j = 0;
          for (size_t i = 0; i < args.Length(); ++i) {
            if (j < argnums.size() && i == argnums[j]) {
              js_args.push_back(
                  TreeUnflatten(env, args[i], primals, strides[j]));
              j++;
            } else {
              js_args.push_back(args[i]);
            }
          }
          // Call the JS function with |js_args|.
          napi_value func = js_func.Value();
          if (napi_make_callback(env, nullptr, func, func,
                                 js_args.size(),
                                 js_args.empty() ? nullptr : &js_args.front(),
                                 &result) != napi_ok) {
            return {};
          }
          // Validate the return value.
          if (!ki::FromNodeTo<mx::array*>(env, result)) {
            if (scalar_func_only) {
              ki::ThrowError(
                  env,
                  error_tag, " The return value of the function whose gradient "
                  "we want to compute should be a scalar array; but ",
                  ki::NodeTypeToString(env, result), " was returned.");
              return {};
            }
            if (!ki::IsArray(env, result)) {
              ki::ThrowError(
                  env,
                  error_tag, " The return value of the function whose gradient "
                  "we want to compute should be either a scalar array or an "
                  "Array with the first value being a scalar array but ",
                  ki::NodeTypeToString(env, result), " was returned.");
              return {};
            }
            auto v = *(ki::FromNodeTo<std::vector<napi_value>>(env, result));
            if (v.empty()) {
              ki::ThrowError(
                  env,
                  error_tag, " The return value of the function whose gradient "
                  "we want to compute should be either a scalar array or a "
                  "non-empty Array. The first value should be a scalar array "
                  "and the rest can be anything. Instead, we got an empty "
                  "Array.");
              return {};
            }
            if (!ki::FromNodeTo<mx::array*>(env, v[0])) {
              ki::ThrowError(
                  env,
                  error_tag, " The return value of the function whose gradient "
                  "we want to compute should be either a scalar array or an "
                  "Array with the first value being a scalar array; but it ",
                  "was a tuple with the first value being of type ",
                  ki::NodeTypeToString(env, v[0]), " .");
              return {};
            }
          }
          // Return flattened results which will be traced.
          return TreeFlatten(env, result);
        }, std::move(gradient_indices));
    // Call the function immediately, because this C++ lambda is actually the
    // result of value_and_grad.
    const auto& [values, gradients] = value_and_grad_func(arrays);
    // Convert gradients to JS value. For array inputs the gradients will be
    // returned, for Array and Object inputs the original arg will be returned
    // with their array properties replaced with corresponding gradients.
    napi_value js_grads = nullptr;
    if (argnums.size() == 1) {
      js_grads = TreeUnflatten(env, args[argnums[0]], gradients, strides[0]);
    } else if (argnums.size() > 0) {
      std::vector<napi_value> grads;
      for (size_t i = 0; i < argnums.size(); ++i) {
        grads.push_back(
            TreeUnflatten(env, args[argnums[i]], gradients, strides[i]));
      }
      js_grads = ki::ToNodeValue(env, grads);
    } else {
      napi_get_null(env, &js_grads);
    }
    return {TreeUnflatten(env, result, values), js_grads};
  };
}

auto ValueAndGrad(ki::Persistent js_func,
                  std::optional<std::variant<int, std::vector<int>>> argnums) {
  return ValueAndGradImpl("[value_and_grad]", false, std::move(js_func),
                          std::move(argnums));
}

std::function<napi_value(ki::Arguments)>
Grad(ki::Persistent js_func,
     std::optional<std::variant<int, std::vector<int>>> argnums) {
  auto func = ValueAndGradImpl("[grad]", true, std::move(js_func),
                               std::move(argnums));
  return [func = std::move(func)](ki::Arguments args) {
    return func(std::move(args)).second;
  };
}

std::function<napi_value(ki::Arguments*)>
VMap(napi_env env,
     napi_value value,
     std::optional<std::variant<int, std::vector<int>>> in_axes,
     std::optional<std::variant<int, std::vector<int>>> out_axes) {
  // Reference the JS function as napi_value only lives at current tick.
  ki::Persistent js_func(env, value);
  // Call vmap with the JS function.
  auto func = mx::vmap(
      [js_func = std::move(js_func)](const std::vector<mx::array>& primals) {
        return ExecuteWithPrimals(js_func.Env(), js_func.Value(), primals);
      },
      ToIntVector(std::move(in_axes.value_or(std::vector<int>()))),
      ToIntVector(std::move(out_axes.value_or(std::vector<int>()))));
  // Return a JS function that converts JS args into primals.
  return [env, func = std::move(func)](ki::Arguments* args) -> napi_value {
    std::vector<mx::array> arrays;
    if (!ReadArgs(args, &arrays))
      return nullptr;
    auto results = func(std::move(arrays));
    if (ki::IsExceptionPending(env))
      return nullptr;
    return UnflattenResults(env, results);
  };
}

std::function<napi_value(ki::Arguments*)>
Compile(napi_env env,
        napi_value value,
        std::optional<bool> shapeless) {
  // Reference the JS function as napi_value only lives at current tick.
  ki::Persistent js_func(env, value);
  std::uintptr_t func_id = reinterpret_cast<std::uintptr_t>(js_func.Id());
  // Call compile with the JS function.
  auto func = mx::detail::compile(
      [js_func = std::move(js_func)](const std::vector<mx::array>& primals) {
        return ExecuteWithPrimals(js_func.Env(), js_func.Value(), primals);
      },
      func_id,
      shapeless.value_or(false));
  // Return a JS function that converts JS args into primals.
  return [env, func = std::move(func)](ki::Arguments* args) -> napi_value {
    std::vector<mx::array> arrays;
    if (!ReadArgs(args, &arrays))
      return nullptr;
    auto results = func(std::move(arrays));
    if (ki::IsExceptionPending(env))
      return nullptr;
    return UnflattenResults(env, results);
  };
}

}  // namespace transforms_ops

void InitTransforms(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "eval", EvalOpWrapper(&mx::eval),
          "asyncEval", EvalOpWrapper(&mx::async_eval),
          "jvp", JVPOpWrapper(&mx::jvp),
          "vjp", JVPOpWrapper(&mx::vjp),
          "valueAndGrad", &transforms_ops::ValueAndGrad,
          "grad", &transforms_ops::Grad,
          "vmap", &transforms_ops::VMap,
          "compile", &transforms_ops::Compile,
          "disableCompile", &mx::disable_compile,
          "enableCompile", &mx::enable_compile);
}
