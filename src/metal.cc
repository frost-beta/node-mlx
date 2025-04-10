#include "src/bindings.h"

void InitMetal(napi_env env, napi_value exports) {
  napi_value metal = ki::CreateObject(env);
  ki::Set(env, exports, "metal", metal);

  ki::Set(env, metal,
          "isAvailable", &mx::metal::is_available,
          "startCapture", &mx::metal::start_capture,
          "stopCapture", &mx::metal::stop_capture,
          "deviceInfo", &mx::metal::device_info);
}
