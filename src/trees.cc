#include "src/array.h"
#include "src/trees.h"

void TreeVisit(napi_env env, napi_value tree,
               const std::function<void(napi_env env,
                                        napi_value tree)>& callback) {
  std::function<void(napi_env env, napi_value value)> recurse;
  recurse = [&callback, &recurse](napi_env env, napi_value value) {
    // Iterate arrays.
    if (ki::IsArray(env, value)) {
      uint32_t length = 0;
      napi_get_array_length(env, value, &length);
      for (uint32_t i = 0; i < length; ++i) {
        napi_value item;
        if (napi_get_element(env, value, i, &item) != napi_ok)
          break;
        recurse(env, item);
      }
      return;
    }
    // Only iterate objects when they do not wrap a native instance.
    void* ptr;
    if (napi_unwrap(env, value, &ptr) != napi_ok) {
      auto m = ki::FromNodeTo<std::map<napi_value, napi_value>>(env, value);
      if (m) {
        for (auto [key, item] : *m)
          recurse(env, item);
        return;
      }
    }
    callback(env, value);
  };

  recurse(env, tree);
}

std::vector<mx::array> TreeFlatten(napi_env env, napi_value tree, bool strict) {
  std::vector<mx::array> flat;
  TreeVisit(env, tree, [strict, &flat](napi_env env, napi_value value) {
    if (auto a = ki::FromNodeTo<mx::array*>(env, value); a) {
      flat.push_back(*a.value());
    } else if (strict) {
      throw std::invalid_argument(
          "[TreeFlatten] The argument should contain only arrays");
    }
  });
  return flat;
}

std::vector<mx::array> TreeFlatten(ki::Arguments* args, bool strict) {
  if (args->Length() == 1)
    return TreeFlatten(args->Env(), (*args)[0], strict);
  std::vector<mx::array> ret;
  for (uint32_t i = 0; i < args->Length(); ++i) {
    std::vector<mx::array> flat = TreeFlatten(args->Env(), (*args)[i], strict);
    std::move(flat.begin(), flat.end(), std::back_inserter(ret));
  }
  return ret;
}
