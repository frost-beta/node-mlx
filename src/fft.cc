#include "src/array.h"
#include "src/stream.h"

// A template converter for ops that accept |n| and |axis|.
inline
std::function<mx::array(const mx::array& a,
                        std::optional<int> n,
                        std::optional<int> axis,
                        mx::StreamOrDevice s)>
FFTOpWrapper(mx::array(*func1)(const mx::array&,
                               int,
                               int,
                               mx::StreamOrDevice),
             mx::array(*func2)(const mx::array&,
                               int,
                               mx::StreamOrDevice)) {
  return [func1, func2](const mx::array& a,
                        std::optional<int> n,
                        std::optional<int> axis,
                        mx::StreamOrDevice s) {
    if (n)
      return func1(a, *n, axis.value_or(-1), s);
    else
      return func2(a, axis.value_or(-1), s);
  };
}

inline
std::function<mx::array(const mx::array& a,
                        std::optional<std::vector<int>> n,
                        std::optional<std::vector<int>> axes,
                        mx::StreamOrDevice s)>
FFTNOpWrapper(const char* name,
              mx::array(*func1)(const mx::array&,
                                const std::vector<int>&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func2)(const mx::array&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func3)(const mx::array&,
                                mx::StreamOrDevice)) {
  return [name, func1, func2, func3](const mx::array& a,
                                     std::optional<std::vector<int>> n,
                                     std::optional<std::vector<int>> axes,
                                     mx::StreamOrDevice s) {
    if (n && axes) {
      return mx::fft::fftn(a, std::move(*n), std::move(*axes), s);
    } else if (axes) {
      return mx::fft::fftn(a, std::move(*axes), s);
    } else if (n) {
      std::ostringstream msg;
      msg << "[" << name << "] "
          << "`axes` should not be `None` if `s` is not `None`.";
      throw std::invalid_argument(msg.str());
    } else {
      return mx::fft::fftn(a, s);
    }
  };
}

inline
std::function<mx::array(const mx::array& a,
                        std::optional<std::vector<int>> n,
                        std::optional<std::vector<int>> axes,
                        mx::StreamOrDevice s)>
FFT2OpWrapper(const char* name,
              mx::array(*func1)(const mx::array&,
                                const std::vector<int>&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func2)(const mx::array&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func3)(const mx::array&,
                                mx::StreamOrDevice)) {
  return [name, func1, func2, func3](const mx::array& a,
                                     std::optional<std::vector<int>> n,
                                     std::optional<std::vector<int>> axes,
                                     mx::StreamOrDevice s) {
    return FFTNOpWrapper(name, func1, func2, func3)(
        a, std::move(n), std::move(axes.value_or(std::vector<int>{-2, -1})), s);
  };
}

void InitFFT(napi_env env, napi_value exports) {
  napi_value fft = ki::CreateObject(env);
  ki::Set(env, exports, "fft", fft);

  ki::Set(env, fft,
          "fft", FFTOpWrapper(&mx::fft::fft,
                              &mx::fft::fft),
          "ifft", FFTOpWrapper(&mx::fft::ifft,
                               &mx::fft::ifft),
          "fft2", FFT2OpWrapper("fft2",
                                &mx::fft::fftn,
                                &mx::fft::fftn,
                                &mx::fft::fftn),
          "ifft2", FFT2OpWrapper("ifft2",
                                 &mx::fft::ifftn,
                                 &mx::fft::ifftn,
                                 &mx::fft::ifftn),
          "fftn", FFTNOpWrapper("fftn",
                                &mx::fft::fftn,
                                &mx::fft::fftn,
                                &mx::fft::fftn),
          "ifftn", FFTNOpWrapper("ifftn",
                                 &mx::fft::ifftn,
                                 &mx::fft::ifftn,
                                 &mx::fft::ifftn),
          "rfft", FFTOpWrapper(&mx::fft::rfft, &mx::fft::rfft),
          "irfft", FFTOpWrapper(&mx::fft::irfft, &mx::fft::irfft),
          "rfft2", FFT2OpWrapper("rfft2",
                                 &mx::fft::rfftn,
                                 &mx::fft::rfftn,
                                 &mx::fft::rfftn),
          "irfft2", FFT2OpWrapper("irfft2",
                                  &mx::fft::irfftn,
                                  &mx::fft::irfftn,
                                  &mx::fft::irfftn),
          "rfftn", FFTNOpWrapper("rfftn",
                                 &mx::fft::rfftn,
                                 &mx::fft::rfftn,
                                 &mx::fft::rfftn),
          "irfftn", FFTNOpWrapper("irfftn",
                                  &mx::fft::irfftn,
                                  &mx::fft::irfftn,
                                  &mx::fft::irfftn));
}
