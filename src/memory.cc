#include "src/bindings.h"

void InitMemory(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "getActiveMemory", &mx::get_active_memory,
          "getPeakMemory", &mx::get_peak_memory,
          "resetPeakMemory", &mx::reset_peak_memory,
          "getCacheMemory", &mx::get_cache_memory,
          "setMemoryLimit", &mx::set_memory_limit,
          "setWiredLimit", &mx::set_wired_limit,
          "setCacheLimit", &mx::set_cache_limit,
          "clearCache", &mx::clear_cache);
}
