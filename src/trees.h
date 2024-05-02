#include "src/bindings.h"

// The callback should return nullptr if it is read-only, returning a value
// would replace the one in original object.
using TreeVisitCallback = std::function<napi_value(napi_env, napi_value)>;
using ListVisitCallback = std::function<napi_value(napi_env, napi_value, bool)>;

// Visiting all leaves in the JS object.
napi_value TreeVisit(napi_env env,
                     napi_value tree,
                     const TreeVisitCallback& visit);

// Visiting the elements in JS object with depth of 1.
napi_value ListVisit(napi_env env,
                     napi_value value,
                     const ListVisitCallback& visit);

// Put all arrays in JS object into a flat vector.
std::vector<mx::array> TreeFlatten(napi_env env,
                                   napi_value tree,
                                   bool strict = false);
std::vector<mx::array> TreeFlatten(ki::Arguments* args, bool strict = false);

// Replace the arrays in JS object with the ones in |arrays|.
napi_value TreeUnflatten(napi_env env,
                         napi_value tree,
                         const std::vector<mx::array>& arrays,
                         size_t index = 0,
                         size_t* new_index = nullptr);

// Like TreeFlatten but also replaces the array in |tree| with a placeholder.
std::vector<mx::array> TreeFlattenWithPlaceholder(napi_env env,
                                                  napi_value tree);
napi_value TreeUnflattenFromPlaceholder(napi_env env,
                                        napi_value tree,
                                        const std::vector<mx::array>& arrays,
                                        size_t index = 0);
