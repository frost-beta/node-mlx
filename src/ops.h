#ifndef SRC_OPS_H_
#define SRC_OPS_H_

#include "src/stream.h"
#include "src/utils.h"

namespace ops {

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

}  // namespace ops

#endif  // SRC_OPS_H_
