#include "src/array.h"
#include "src/trees.h"

napi_value TreeVisit(napi_env env,
                     napi_value tree,
                     const TreeVisitCallback& visit) {
  TreeVisitCallback recurse;
  recurse = [&visit, &recurse](napi_env env, napi_value value) -> napi_value {
    // Iterate arrays.
    if (ki::IsArray(env, value)) {
      uint32_t length = 0;
      napi_get_array_length(env, value, &length);
      for (uint32_t i = 0; i < length; ++i) {
        napi_value item;
        if (napi_get_element(env, value, i, &item) != napi_ok)
          break;
        napi_value result = recurse(env, item);
        if (result)
          napi_set_element(env, value, i, result);
      }
      return nullptr;
    }
    // Only iterate objects when they do not wrap a native instance.
    void* ptr;
    if (napi_unwrap(env, value, &ptr) != napi_ok) {
      auto m = ki::FromNodeTo<std::map<napi_value, napi_value>>(env, value);
      if (m) {
        for (auto [key, item] : *m) {
          napi_value result = recurse(env, item);
          if (result)
            napi_set_property(env, value, key, result);
        }
        return nullptr;
      }
    }
    return visit(env, value);
  };

  return recurse(env, tree);
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
    return napi_value();
  });
  return flat;
}

std::vector<mx::array> TreeFlatten(ki::Arguments* args, bool strict) {
  // Fast route for single element.
  if (args->Length() == 1)
    return TreeFlatten(args->Env(), (*args)[0], strict);
  // Iterate args and put merge results into one vector.
  std::vector<mx::array> ret;
  for (size_t i = 0; i < args->Length(); ++i) {
    std::vector<mx::array> flat = TreeFlatten(args->Env(), (*args)[i], strict);
    // Move concatenation.
    std::move(flat.begin(), flat.end(), std::back_inserter(ret));
  }
  return ret;
}

napi_value TreeUnflatten(napi_env env,
                         napi_value tree,
                         const std::vector<mx::array>& arrays,
                         size_t index) {
  napi_value result = TreeVisit(
      env, tree, [&arrays, &index](napi_env env, napi_value value) {
    if (ki::FromNodeTo<mx::array*>(env, value))
      return ki::ToNodeValue(env, arrays[index++]);
    return napi_value();
  });
  return result ? result : tree;
}
