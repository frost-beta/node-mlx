#include "src/array.h"

// A template converter for ops that accept infinite |array|s.
inline
std::function<void(ki::Arguments* args)>
EvalOpWrapper(void(*func)(std::vector<mx::array>)) {
  return [func](ki::Arguments* args) {
    std::vector<mx::array> arrays;
    for (size_t i = 0; i < args->Length(); ++i) {
      std::optional<mx::array> a = args->GetNext<mx::array>();
      if (!a) {
        args->ThrowError("array");
        return;
      }
      arrays.push_back(std::move(*a));
    }
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
      // Convert primals to JS values.
      std::vector<napi_value> args;
      for (const mx::array& primal : primals)
        args.push_back(ki::ToNodeValue(env, primal));
      // Call the JS function with primals.
      napi_value result = nullptr;
      napi_make_callback(env, nullptr, js_func, js_func,
                         args.size(), args.empty() ? nullptr : &args.front(),
                         &result);
      // Convert result to vector.
      if (auto a = ki::FromNodeTo<mx::array>(env, result); a)
        return std::vector<mx::array>{std::move(*a)};
      if (auto v = ki::FromNodeTo<std::vector<mx::array>>(env, result); v)
        return std::move(*v);
      throw new std::runtime_error("function does not return mx.array or "
                                   "an Array of mx.array");
    };
    return func(vfunc, primals, tangents);
  };
}

void InitTransforms(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "eval", EvalOpWrapper(&mx::eval),
          "asyncEval", EvalOpWrapper(&mx::async_eval),
          "jvp", JVPOpWrapper(&mx::jvp),
          "vjp", JVPOpWrapper(&mx::vjp));
}
