#include "src/array.h"
#include "src/stream.h"

namespace io {

std::variant<mx::array,
             std::unordered_map<std::string, mx::array>,
             mx::SafetensorsLoad,
             mx::GGUFLoad>
Load(std::string file,
     std::optional<std::string> format,
     std::optional<bool> return_metadata_arg,
     mx::StreamOrDevice s) {
  if (!format) {
    size_t ext = file.find_last_of('.');
    if (ext == std::string::npos) {
      throw std::invalid_argument(
          "[load] Could not infer file format from extension");
    }
    format = file.substr(ext + 1);
  }

  bool return_metadata = return_metadata_arg.value_or(false);
  if (return_metadata && (*format == "npy" || *format == "npz")) {
    throw std::invalid_argument(
        "[load] metadata not supported for format " + *format);
  }

  if (*format == "safetensors") {
    auto [dict, metadata] = mx::load_safetensors(std::move(file), s);
    if (return_metadata)
      return std::make_pair(dict, metadata);
    return dict;
  } else if (*format == "npy") {
    return mx::load(std::move(file), s);
  } else if (*format == "gguf") {
    auto [weights, metadata] = mx::load_gguf(std::move(file), s);
    if (return_metadata)
      return std::make_pair(weights, metadata);
    else
      return weights;
  } else {
    throw std::invalid_argument("[load] Unknown file format " + *format);
  }
}

void SaveSafetensors(
    std::string file,
    std::unordered_map<std::string, mx::array> arrays,
    std::optional<std::unordered_map<std::string, std::string>> metadata) {
  return mx::save_safetensors(
      std::move(file),
      std::move(arrays),
      std::move(metadata.value_or(decltype(metadata)::value_type())));
}

void SaveGGUF(
    std::string file,
    std::unordered_map<std::string, mx::array> arrays,
    std::optional<std::unordered_map<std::string, mx::GGUFMetaData>> metadata) {
  return mx::save_gguf(
      std::move(file),
      std::move(arrays),
      std::move(metadata.value_or(decltype(metadata)::value_type())));
}

}  // namespace io

void InitIO(napi_env env, napi_value exports) {
  ki::Set(env, exports,
          "load", &io::Load,
          "save", static_cast<void(*)(std::string, mx::array)>(&mx::save),
          "saveSafetensors", &io::SaveSafetensors,
          "saveGGUF", &io::SaveGGUF);
}
