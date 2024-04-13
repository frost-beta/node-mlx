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

}  // namespace ops

#endif  // SRC_OPS_H_
