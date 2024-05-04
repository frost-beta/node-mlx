#include "src/bindings.h"

namespace metal_ops {

int SetMemoryLimit(int limit, std::optional<bool> relaxed) {
  return mx::metal::set_memory_limit(limit, relaxed.value_or(true));
}

}  // namespace metal_ops

void InitMetal(napi_env env, napi_value exports) {
  napi_value metal = ki::CreateObject(env);
  ki::Set(env, exports, "metal", metal);

  ki::Set(env, metal,
          "isAvailable", &mx::metal::is_available,
          "getActiveMemory", &mx::metal::get_active_memory,
          "getPeakMemory", &mx::metal::get_peak_memory,
          "resetPeakMemory", &mx::metal::reset_peak_memory,
          "getCacheMemory", &mx::metal::get_cache_memory,
          "setMemoryLimit", &metal_ops::SetMemoryLimit,
          "clearCache", &mx::metal::clear_cache,
          "setCacheLimit", &mx::metal::set_cache_limit,
          "startCapture", &mx::metal::start_capture,
          "stopCapture", &mx::metal::stop_capture,
          "deviceInfo", &mx::metal::device_info);
}
