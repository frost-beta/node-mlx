#include "src/bindings.h"

void TreeVisit(napi_env env, napi_value tree,
               const std::function<void(napi_env env,
                                        napi_value tree)>& callback);

std::vector<mx::array> TreeFlatten(napi_env env, napi_value tree,
                                   bool strict = false);
std::vector<mx::array> TreeFlatten(ki::Arguments* args, bool strict = false);
