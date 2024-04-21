#ifndef SRC_OPS_H_
#define SRC_OPS_H_

#include "src/stream.h"
#include "src/utils.h"

// A template converter for ops that accept |axis|.
inline
std::function<mx::array(const mx::array& a,
                        IntOrVector axis,
                        mx::StreamOrDevice s)>
DimOpWrapper(mx::array(*func)(const mx::array&,
                              const std::vector<int>&,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                IntOrVector axis,
                mx::StreamOrDevice s) {
    return func(a, GetReduceAxes(std::move(axis), a.ndim()), s);
  };
}

// A template converter for ops that accept |axis| and |keepdims|.
inline
std::function<mx::array(const mx::array& a,
                        ki::Arguments* args)>
DimOpWrapper(mx::array(*func)(const mx::array&,
                              const std::vector<int>&,
                              bool,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                ki::Arguments* args) {
    auto axis = args->TryGetNext<IntOrVector>().value_or(std::monostate());
    auto keepdims = args->TryGetNext<bool>().value_or(false);
    auto s = args->TryGetNext<mx::StreamOrDevice>().value_or(std::monostate());
    return func(a, GetReduceAxes(std::move(axis), a.ndim()), keepdims, s);
  };
}

// A template converter for |cum| ops.
inline
std::function<mx::array(const mx::array& a,
                        std::optional<int> axis,
                        std::optional<bool> reverse,
                        std::optional<bool> inclusive,
                        mx::StreamOrDevice s)>
CumOpWrapper(mx::array(*func)(const mx::array&,
                              int,
                              bool,
                              bool,
                              mx::StreamOrDevice)) {
  return [func](const mx::array& a,
                std::optional<int> axis,
                std::optional<bool> reverse,
                std::optional<bool> inclusive,
                mx::StreamOrDevice s) {
    bool r = reverse.value_or(false);
    bool i = reverse.value_or(true);
    if (axis)
      return func(a, *axis, r, i, s);
    else
      return func(mx::reshape(a, {-1}, s), 0, r, i, s);
  };
}

namespace ops {

// Following op implementations are also used by array bindings.
mx::array Flatten(const mx::array& a,
                  std::optional<int> start_axis,
                  std::optional<int> end_axis,
                  mx::StreamOrDevice s);
mx::array Squeeze(const mx::array& a,
                  IntOrVector axis,
                  mx::StreamOrDevice s);
mx::array Transpose(const mx::array& a,
                    std::optional<std::vector<int>> axis,
                    mx::StreamOrDevice s);
mx::array Var(const mx::array& a,
              IntOrVector axis,
              std::optional<bool> keepdims,
              std::optional<int> ddof,
              mx::StreamOrDevice s);
std::vector<mx::array> Split(const mx::array& a,
                             std::variant<int, std::vector<int>> indices,
                             std::optional<int> axis,
                             mx::StreamOrDevice s);
mx::array ArgMin(const mx::array& a,
                 std::optional<int> axis,
                 std::optional<bool> keepdims,
                 mx::StreamOrDevice s);
mx::array ArgMax(const mx::array& a,
                 std::optional<int> axis,
                 std::optional<bool> keepdims,
                 mx::StreamOrDevice s);
mx::array Round(const mx::array& a,
                std::optional<int> decimals,
                mx::StreamOrDevice s);
mx::array Diagonal(const mx::array& a,
                   std::optional<int> offset,
                   std::optional<int> axis1,
                   std::optional<int> axis2,
                   mx::StreamOrDevice s);
mx::array Diag(const mx::array& a,
               std::optional<int> k,
               mx::StreamOrDevice s);

}  // namespace ops

#endif  // SRC_OPS_H_
