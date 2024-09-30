#include "src/array.h"
#include "src/ops.h"
#include "src/trees.h"

// Needed for detail::compile.
#include "mlx/transforms_impl.h"

namespace {

// Use large prime numbers to represent non-constant JS elements.
constexpr uint64_t kArrayIdentifier = 18446744073709551557UL;
constexpr uint64_t kListIdentifier = 18446744073709551533UL;
constexpr uint64_t kDictIdentifier = 18446744073709551521UL;
constexpr uint64_t kArgIdentifier = 18446744073709551437UL;

// Shares data between workers.
struct WorkerData {
  napi_env env = nullptr;
  napi_async_work work = nullptr;
  napi_deferred deffered = nullptr;
  std::vector<mx::array> arrays;

  ~WorkerData() {
    if (deffered) {
      napi_reject_deferred(env, deffered,
                           ki::ToNodeValue(env, "Worker failed."));
    }
    if (work) {
      napi_delete_async_work(env, work);
    }
  }
};

// Used by compiled function to transfer JS results to callers.
auto& CompiledFunctionRelay() {
  static std::map<void*, ki::Persistent> relay;
  return relay;
}

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

// Converts a string to a uint64_t, does not need to be reliable.
uint64_t StrToConstant(std::string_view str) {
  uint64_t r = 1;
  size_t length = std::min<size_t>(20, str.size());
  for (size_t i = 0; i < length; ++i)
    r *= str[i];
  return r;
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

std::function<std::pair<napi_value, napi_value>(ki::Arguments)>
ValueAndGradImpl(const char* error_tag,
                 bool scalar_func_only,
                 ki::Persistent js_func,
                 std::optional<std::variant<int, std::vector<int>>> arg) {
  // Sanitize |argnums|.
  std::vector<int> argnums = PutIntoVector(std::move(arg.value_or(0)));
  if (argnums.size() > 0) {
    std::sort(argnums.begin(), argnums.end());
    if (argnums[0] < 0) {
      ki::ThrowError(js_func.Env(),
                     error_tag, " Can't compute the gradient of negative "
                     "argument index ", argnums[0]);
      return nullptr;
    }
  }
  // Return a wrapper that does value_and_grad lazily instead of calling the
  // function directly. This simplifies the implementation, and is fine since
  // the result is usually called immediately and only once.
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
                     "called with only ", args.Length(),
                     " positional arguments.");
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
    } else if (argnums.size() > 1) {
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

}  // namespace

namespace transforms_ops {

napi_value EvalInWorker(ki::Arguments* args) {
  std::unique_ptr<WorkerData> data = std::make_unique<WorkerData>();
  data->env = args->Env();
  if (napi_create_async_work(
      data->env,
      nullptr,
      ki::ToNodeValue(data->env, "evalInWorker"),
      [](napi_env env, void* hint) {
        auto* data = static_cast<WorkerData*>(hint);
        // Call actual eval, do not move the arrays otherwise the underlying
        // data might end up getting freed in worker, which causes race
        // conditions.
        mx::eval(data->arrays);
      },
      [](napi_env env, napi_status status, void* hint) {
        auto* data = static_cast<WorkerData*>(hint);
        // Resolve promise and release everything on complete.
        napi_resolve_deferred(env, data->deffered, ki::Undefined(env));
        data->deffered = nullptr;
        delete data;
      },
      data.get(),
      &data->work) != napi_ok) {
    args->ThrowError("Failed to create async work");
    return nullptr;
  }
  // Create the returned promise.
  napi_value result;
  if (napi_create_promise(data->env, &data->deffered, &result) != napi_ok) {
    args->ThrowError("Failed to create promise");
    return nullptr;
  }
  // Start the work.
  data->arrays = TreeFlatten(args);
  if (napi_queue_async_work(data->env, data->work) != napi_ok) {
    args->ThrowError("Failed to queue async work");
    return nullptr;
  }
  // Leak the data, which will be freed in complete handler.
  data.release();
  return result;
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
VMap(ki::Persistent js_func,
     std::optional<std::variant<int, std::vector<int>>> in_axes,
     std::optional<std::variant<int, std::vector<int>>> out_axes) {
  // Call vmap with the JS function.
  auto func = mx::vmap(
      [js_func = std::move(js_func)](const std::vector<mx::array>& primals) {
        return ExecuteWithPrimals(js_func.Env(), js_func.Value(), primals);
      },
      PutIntoVector(std::move(in_axes.value_or(std::vector<int>()))),
      PutIntoVector(std::move(out_axes.value_or(std::vector<int>()))));
  // Return a JS function that converts JS args into primals.
  return [func = std::move(func)](ki::Arguments* args) -> napi_value {
    std::vector<mx::array> arrays;
    if (!ReadArgs(args, &arrays))
      return nullptr;
    auto results = func(std::move(arrays));
    if (ki::IsExceptionPending(args->Env()))
      return nullptr;
    return UnflattenResults(args->Env(), results);
  };
}

std::function<napi_value(ki::Arguments)>
Compile(ki::Persistent js_func, std::optional<bool> shapeless) {
  // The |erase_compilation_cache| will be called when |freer| is destroyed, and
  // the |freer|'s lifetime is managed by the returned compiled function.
  // i.e. The compilation cache will be freed when compiled function is garbage
  // collected in JS.
  auto erase_compilation_cache = [](void* func_id) {
    CompiledFunctionRelay().erase(func_id);
    mx::detail::compile_erase(reinterpret_cast<std::uintptr_t>(func_id));
  };
  std::shared_ptr<void> freer(js_func.Id(), erase_compilation_cache);
  // Return a wrapper that calls mx::compile lazily instead of calling the
  // function directly, because the native compiled function "freezes" all the
  // non-array args and for the JS function that takes arbitrary args to work
  // we have to re-compile the function for each call that passes different set
  // of constants.
  return [js_func = std::move(js_func),
          freer = std::move(freer),
          shapeless](ki::Arguments args)
        -> napi_value {
    // Iterate the args to record its tree structure, will be used as part of
    // the keys for deciding whether to re-compile the function.
    std::vector<mx::array> inputs;
    std::vector<uint64_t> records;
    ListVisitCallback recurse;
    recurse = [&recurse, &records, &inputs](napi_env env,
                                            napi_value value,
                                            bool is_leaf) {
      napi_valuetype type = napi_undefined;
      napi_typeof(env, value, &type);
      if (is_leaf) {
        if (auto a = ki::FromNodeTo<mx::array*>(env, value); a) {
          records.push_back(kArrayIdentifier);
          inputs.push_back(*a.value());
        } else if (type == napi_boolean) {
          records.push_back(*ki::FromNodeTo<bool>(env, value) ? 1 : 0);
        } else if (type == napi_number) {
          // Re-represent the number with uint64_t. Do not use static_cast as
          // float numbers like 0.1/0.01 will all become 0.
          double f = *ki::FromNodeTo<double>(env, value);
          records.push_back(*reinterpret_cast<uint64_t*>(&f));
        } else if (type == napi_string) {
          records.push_back(
              StrToConstant(*ki::FromNodeTo<std::string>(env, value)));
        } else {
          throw new std::invalid_argument(
              "[compile] Function arguments must be recordss of arrays or "
              "constants (booleans, numbers, or strings).");
        }
      } else {
        if (ki::IsArray(env, value))
          records.push_back(kListIdentifier);
        else if (type == napi_object)
          records.push_back(kDictIdentifier);
        ListVisit(env, value, recurse);
      }
      return nullptr;
    };
    for (size_t i = 0; i < args.Length(); ++i) {
      records.push_back(kArgIdentifier);
      ListVisit(args.Env(), args[i], recurse);
    }

    // Call compile with the JS function.
    auto func = mx::detail::compile(
        // Note that this function is only called when there is cache-miss.
        [&js_func, &args, &inputs](const std::vector<mx::array>& primals) {
          // Read the args into |js_args| vector, and replace the arrays in it
          // with the traced |primals|.
          std::vector<napi_value> js_args;
          size_t index = 0;
          for (size_t i = 0; i < args.Length(); ++i) {
            js_args.push_back(
                TreeUnflatten(args.Env(), args[i], primals, index, &index));
          }
          // Call the JS function with |js_args|.
          napi_value func = js_func.Value();
          napi_value result;
          if (napi_make_callback(args.Env(), nullptr, func, func,
                                 js_args.size(),
                                 js_args.empty() ? nullptr : &js_args.front(),
                                 &result) != napi_ok) {
            return std::vector<mx::array>();
          }
          // Get the array elements from |result|, and replace them with
          // placeholders in the |replaced| object.
          auto [outputs, replaced] = TreeFlattenWithPlaceholder(args.Env(),
                                                                result);
          // As this function is not called after it is compiled, we can not
          // transfer the JS result to caller. Instead we are saving the result
          // of first call when it was being compiled, and reuse the object as
          // result for following calls by replacing the array elements with
          // the ones from new results.
          CompiledFunctionRelay().emplace(js_func.Id(),
                                          ki::Persistent(args.Env(), replaced));
          return outputs;
        },
        reinterpret_cast<std::uintptr_t>(js_func.Id()),
        shapeless.value_or(false),
        std::move(records));
    // Call the compiled function.
    std::vector<mx::array> outputs = func(inputs);
    if (ki::IsExceptionPending(args.Env()))
      return nullptr;
    // Get the placeholder result object for the compiled function.
    auto it = CompiledFunctionRelay().find(js_func.Id());
    if (it == CompiledFunctionRelay().end()) {
      ki::ThrowError(args.Env(), "The compiled function did not provide any "
                                 "result, was it failed?");
      return nullptr;
    }
    // Replace the placeholders in the object with actual array results.
    napi_value result = it->second.Value();
    return TreeUnflattenFromPlaceholder(args.Env(), result, outputs);
  };
}

}  // namespace transforms_ops

void InitTransforms(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "eval", EvalOpWrapper(&mx::eval),
          "asyncEval", EvalOpWrapper(&mx::async_eval),
          "evalInWorker", &transforms_ops::EvalInWorker,
          "jvp", JVPOpWrapper(&mx::jvp),
          "vjp", JVPOpWrapper(&mx::vjp),
          "valueAndGrad", &transforms_ops::ValueAndGrad,
          "grad", &transforms_ops::Grad,
          "vmap", &transforms_ops::VMap,
          "compile", &transforms_ops::Compile,
          "disableCompile", &mx::disable_compile,
          "enableCompile", &mx::enable_compile);
}
