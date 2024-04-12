#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_

#include <kizunapi.h>

// In js land the objects are always stored as pointers, when a value is needed
// from C++ land, we do a copy.
template<typename T>
inline std::optional<T> NodeObjToCppValue(napi_env env, napi_value value) {
  std::optional<T*> ptr = ki::FromNode<T*>(env, value);
  if (!ptr)
    return std::nullopt;
  return *ptr.value();
}

#endif  // SRC_UTIL_H_
