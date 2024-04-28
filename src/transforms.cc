#include "src/array.h"
#include "src/ops.h"

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
    std::vector<mx::array> arrays;
    if (ReadArgs(args, &arrays))
      func(std::move(arrays));
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

std::function<std::pair<napi_value, napi_value>(ki::Arguments*)>
ValueAndGrad(napi_env env,
             napi_value value,
             std::optional<std::variant<int, std::vector<int>>> argnums) {
  // Reference the JS function as napi_value only lives at current tick.
  ki::Persistent js_func(env, value);
  // Call value_and_grad with the JS function.
  auto func = mx::value_and_grad(
      [js_func = std::move(js_func)](const std::vector<mx::array>& primals) {
        return ExecuteWithPrimals(js_func.Env(), js_func.Value(), primals);
      }, ToIntVector(std::move(argnums.value_or(0))));
  // Return a JS function that converts JS args into primals.
  return [env, func = std::move(func)](ki::Arguments* args)
        -> std::pair<napi_value, napi_value> {
    std::vector<mx::array> arrays;
    if (!ReadArgs(args, &arrays))
      return {nullptr, nullptr};
    auto results = func(std::move(arrays));
    if (ki::IsExceptionPending(env))
      return {nullptr, nullptr};
    return {UnflattenResults(env, results.first),
            UnflattenResults(env, results.second)};
  };
}

std::function<napi_value(ki::Arguments*)>
Grad(napi_env env,
     napi_value js_func,
     std::optional<std::variant<int, std::vector<int>>> argnums) {
  auto func = ValueAndGrad(env, js_func, std::move(argnums));
  return [func = std::move(func)](ki::Arguments* args) {
    return func(args).second;
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
