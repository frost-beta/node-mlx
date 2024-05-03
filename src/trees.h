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

// Similiar to TreeVisit, but instead of modifying the passed tree directly,
// a new tree will be returned.
napi_value TreeMap(napi_env env,
                   napi_value tree,
                   const TreeVisitCallback& visit);

// Put all arrays in JS object into a flat vector.
std::vector<mx::array> TreeFlatten(napi_env env,
                                   napi_value tree,
                                   bool strict = false);
std::vector<mx::array> TreeFlatten(ki::Arguments* args, bool strict = false);

// Return a new tree by replacing the arrays with the passed |arrays|.
napi_value TreeUnflatten(napi_env env,
                         napi_value tree,
                         const std::vector<mx::array>& arrays,
                         size_t index = 0,
                         size_t* new_index = nullptr);

// Like TreeFlatten but return a new tree by replacing the array in |tree| with
// a placeholder.
std::pair<std::vector<mx::array>, napi_value> TreeFlattenWithPlaceholder(
    napi_env env,
    napi_value tree);
napi_value TreeUnflattenFromPlaceholder(napi_env env,
                                        napi_value tree,
                                        const std::vector<mx::array>& arrays,
                                        size_t index = 0);
