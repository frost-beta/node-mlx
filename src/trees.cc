#include "src/array.h"
#include "src/trees.h"

namespace {

// An empty object that is different from any other value.
struct Placeholder {};

// Its address is used as key.
Placeholder kPlaceholderTag;

}  // namespace

namespace ki {

template<>
struct Type<Placeholder> {
  static constexpr const char* name = "Placeholder";
  static inline napi_status ToNode(napi_env env,
                                   const Placeholder& value,
                                   napi_value* result) {
    return napi_create_external(env, &kPlaceholderTag, nullptr, nullptr,
                                result);
  }
  static inline std::optional<Placeholder> FromNode(napi_env env,
                                                    napi_value value) {
    void* result;
    if (napi_get_value_external(env, value, &result) == napi_ok &&
        result == &kPlaceholderTag) {
      return Placeholder();
    }
    return std::nullopt;
  }
};

}  // namespace ki

napi_value TreeVisit(napi_env env,
                     napi_value tree,
                     const TreeVisitCallback& visit) {
  ListVisitCallback recurse;
  recurse = [&recurse, &visit](napi_env env, napi_value value, bool is_leaf) {
    if (is_leaf)
      return visit(env, value);
    else
      return ListVisit(env, value, recurse);
  };
  return ListVisit(env, tree, recurse);
}

napi_value ListVisit(napi_env env,
                     napi_value value,
                     const ListVisitCallback& visit) {
  // Iterate arrays.
  if (ki::IsArray(env, value)) {
    uint32_t length = 0;
    napi_get_array_length(env, value, &length);
    for (uint32_t i = 0; i < length; ++i) {
      napi_value item;
      if (napi_get_element(env, value, i, &item) != napi_ok)
        break;
      napi_value result = visit(env, item, false);
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
        napi_value result = visit(env, item, false);
        if (result)
          napi_set_property(env, value, key, result);
      }
      return nullptr;
    }
  }
  return visit(env, value, true);
}

napi_value TreeMap(napi_env env,
                   napi_value tree,
                   const TreeVisitCallback& visit) {
  TreeVisitCallback recurse;
  recurse = [&recurse, &visit](napi_env env, napi_value value) {
    // Iterate arrays.
    if (ki::IsArray(env, value)) {
      uint32_t length = 0;
      napi_get_array_length(env, value, &length);
      napi_value new_array = nullptr;
      napi_create_array_with_length(env, length, &new_array);
      for (uint32_t i = 0; i < length; ++i) {
        napi_value item;
        if (napi_get_element(env, value, i, &item) != napi_ok)
          break;
        napi_value result = visit(env, item);
        if (result)
          napi_set_element(env, new_array, i, result);
      }
      return new_array;
    }
    // Only iterate objects when they do not wrap a native instance.
    void* ptr;
    if (napi_unwrap(env, value, &ptr) != napi_ok) {
      auto m = ki::FromNodeTo<std::map<napi_value, napi_value>>(env, value);
      if (m) {
        napi_value new_dict = nullptr;
        napi_create_object(env, &new_dict);
        for (auto [key, item] : *m) {
          napi_value result = visit(env, item);
          if (result)
            napi_set_property(env, new_dict, key, result);
        }
        return new_dict;
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
                         size_t index,
                         size_t* new_index) {
  napi_value result = TreeMap(
      env, tree,
      [&arrays, &index](napi_env env, napi_value value) -> napi_value {
    if (ki::FromNodeTo<mx::array*>(env, value)) {
      return ki::ToNodeValue(env, arrays[index++]);
    } else {
      return value;
    }
  });
  if (new_index)
    *new_index = index;
  return result ? result : tree;
}

std::vector<mx::array> TreeFlattenWithPlaceholder(napi_env env,
                                                  napi_value tree) {
  std::vector<mx::array> flat;
  TreeVisit(env, tree, [&flat](napi_env env, napi_value value) -> napi_value {
    if (auto a = ki::FromNodeTo<mx::array*>(env, value); a) {
      flat.push_back(*a.value());
      return ki::ToNodeValue(env, Placeholder());
    } else {
      return value;
    }
  });
  return flat;
}

napi_value TreeUnflattenFromPlaceholder(napi_env env,
                                        napi_value tree,
                                        const std::vector<mx::array>& arrays,
                                        size_t index) {
  napi_value result = TreeVisit(
      env, tree,
      [&arrays, &index](napi_env env, napi_value value) -> napi_value {
    if (ki::FromNodeTo<Placeholder>(env, value)) {
      return ki::ToNodeValue(env, arrays[index++]);
    } else {
      return value;
    }
  });
  return result ? result : tree;
}
