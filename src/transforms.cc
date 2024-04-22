#include "src/array.h"

namespace {

// Read args into a vector of arrays.
bool ReadArgsToArrays(ki::Arguments* args, std::vector<mx::array>* results) {
  for (size_t i = 0; i < args->Length(); ++i) {
    std::optional<mx::array> a = args->GetNext<mx::array>();
    if (!a) {
      args->ThrowError("array");
      return false;
    }
    results->push_back(std::move(*a));
  }
  return true;
}

// Execute JS function with primals.
napi_value ExecuteWithPrimals(napi_env env,
                              napi_value js_func,
                              const std::vector<mx::array>& primals) {
  // Convert primals to JS values.
  std::vector<napi_value> args;
  for (const mx::array& primal : primals)
    args.push_back(ki::ToNodeValue(env, primal));
  // Call the JS function with primals.
  napi_value result = nullptr;
  napi_make_callback(env, nullptr, js_func, js_func,
                     args.size(), args.empty() ? nullptr : &args.front(),
                     &result);
  return result;
}

// A template converter for ops that accept infinite |array|s.
inline
std::function<void(ki::Arguments* args)>
EvalOpWrapper(void(*func)(std::vector<mx::array>)) {
  return [func](ki::Arguments* args) {
    std::vector<mx::array> arrays;
    if (ReadArgsToArrays(args, &arrays))
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
      // Call the JS function with primals.
      napi_value result = ExecuteWithPrimals(env, js_func, primals);
      // Convert result to vector.
      if (auto a = ki::FromNodeTo<mx::array*>(env, result); a)
        return std::vector<mx::array>{*a.value()};
      if (auto v = ki::FromNodeTo<std::vector<mx::array>>(env, result); v)
        return std::move(*v);
      throw new std::runtime_error("function does not return mx.array or "
                                   "an Array of mx.array");
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
  // Get the indices of gradients.
  std::vector<int> gradient_indices = ToIntVector(
      std::move(argnums.value_or(std::vector<int>{0})));
  bool multi_gradients = gradient_indices.size() > 1;
  // Call value_and_grad with the JS function.
  auto func = mx::value_and_grad(
      [env, js_func = std::move(js_func)](
          const std::vector<mx::array>& primals) {
        // Call the JS function with primals.
        napi_value result = ExecuteWithPrimals(env, js_func.Value(), primals);
        // Convert result to vector.
        if (auto a = ki::FromNodeTo<mx::array*>(env, result); a)
          return std::vector<mx::array>{*a.value()};
        if (auto v = ki::FromNodeTo<std::vector<mx::array>>(env, result); v)
          return std::move(*v);
        throw new std::runtime_error("function does not return mx.array or "
                                     "an Array of mx.array or scalar");
      }, std::move(gradient_indices));
  // Return a JS function that converts JS args into primals.
  return [env, func = std::move(func), multi_gradients](ki::Arguments* args) {
    std::pair<napi_value, napi_value> ret;
    std::vector<mx::array> arrays;
    if (!ReadArgsToArrays(args, &arrays))
      return ret;
    auto results = func(std::move(arrays));
    // Unflatten the results.
    if (results.first.size() > 1)
      ret.first = ki::ToNodeValue(env, results.first);
    else
      ret.first = ki::ToNodeValue(env, results.first[0]);
    if (multi_gradients)
      ret.second = ki::ToNodeValue(env, results.second);
    else
      ret.second = ki::ToNodeValue(env, results.second[0]);
    return ret;
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

}  // namespace transforms_ops

void InitTransforms(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "eval", EvalOpWrapper(&mx::eval),
          "asyncEval", EvalOpWrapper(&mx::async_eval),
          "jvp", JVPOpWrapper(&mx::jvp),
          "vjp", JVPOpWrapper(&mx::vjp),
          "valueAndGrad", &transforms_ops::ValueAndGrad,
          "grad", &transforms_ops::Grad);
}
