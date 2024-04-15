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
FFTNOpWrapper(mx::array(*func1)(const mx::array&,
                                const std::vector<int>&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func2)(const mx::array&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func3)(const mx::array&,
                                mx::StreamOrDevice)) {
  return [func1, func2, func3](const mx::array& a,
                               std::optional<std::vector<int>> n,
                               std::optional<std::vector<int>> axes,
                               mx::StreamOrDevice s) {
    if (n && axes) {
      return mx::fft::fftn(a, std::move(*n), std::move(*axes), s);
    } else if (axes) {
      return mx::fft::fftn(a, std::move(*axes), s);
    } else if (n) {
      std::vector<int> all(n->size());
      std::iota(all.begin(), all.end(), -n.value().size());
      return mx::fft::fftn(a, std::move(*n), std::move(all), s);
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
FFT2OpWrapper(mx::array(*func1)(const mx::array&,
                                const std::vector<int>&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func2)(const mx::array&,
                                const std::vector<int>&,
                                mx::StreamOrDevice),
              mx::array(*func3)(const mx::array&,
                                mx::StreamOrDevice)) {
  return [func1, func2, func3](const mx::array& a,
                               std::optional<std::vector<int>> n,
                               std::optional<std::vector<int>> axes,
                               mx::StreamOrDevice s) {
    return FFTNOpWrapper(func1, func2, func3)(
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
          "fft2", FFT2OpWrapper(&mx::fft::fftn,
                                &mx::fft::fftn,
                                &mx::fft::fftn),
          "ifft2", FFT2OpWrapper(&mx::fft::ifftn,
                                 &mx::fft::ifftn,
                                 &mx::fft::ifftn),
          "fftn", FFTNOpWrapper(&mx::fft::fftn,
                                &mx::fft::fftn,
                                &mx::fft::fftn),
          "ifftn", FFTNOpWrapper(&mx::fft::ifftn,
                                 &mx::fft::ifftn,
                                 &mx::fft::ifftn),
          "rfft", FFTOpWrapper(&mx::fft::rfft, &mx::fft::rfft),
          "irfft", FFTOpWrapper(&mx::fft::irfft, &mx::fft::irfft),
          "rfft2", FFT2OpWrapper(&mx::fft::rfftn,
                                 &mx::fft::rfftn,
                                 &mx::fft::rfftn),
          "irfft2", FFT2OpWrapper(&mx::fft::irfftn,
                                  &mx::fft::irfftn,
                                  &mx::fft::irfftn),
          "rfftn", FFTNOpWrapper(&mx::fft::rfftn,
                                 &mx::fft::rfftn,
                                 &mx::fft::rfftn),
          "irfftn", FFTNOpWrapper(&mx::fft::irfftn,
                                  &mx::fft::irfftn,
                                  &mx::fft::irfftn));
}
