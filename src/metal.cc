#include "src/bindings.h"

void InitMetal(napi_env env, napi_value exports) {
  napi_value metal = ki::CreateObject(env);
  ki::Set(env, exports, "metal", metal);

  ki::Set(env, metal,
          "isAvailable", &mx::metal::is_available,
          "getActiveMemory", &mx::metal::get_active_memory,
          "getPeakMemory", &mx::metal::get_peak_memory,
          "getCacheMemory", &mx::metal::get_cache_memory,
          "setMemoryLimit", &mx::metal::set_memory_limit,
          "setCacheLimit", &mx::metal::set_cache_limit,
          "startCapture", &mx::metal::start_capture,
          "stopCapture", &mx::metal::stop_capture);
}
