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

void InitTransforms(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "eval", EvalOpWrapper(&mx::eval),
          "asyncEval", EvalOpWrapper(&mx::async_eval));
}
