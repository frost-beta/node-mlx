#include "src/array.h"
#include "src/stream.h"

namespace random_ops {

mx::array Split(const mx::array& a,
                std::optional<int> num,
                mx::StreamOrDevice s) {
  return mx::random::split(a, num.value_or(2), s);
}

mx::array Uniform(const mx::array& low,
                  const mx::array& high,
                  std::optional<std::vector<int>> shape,
                  std::optional<mx::Dtype> dtype,
                  std::optional<mx::array> key,
                  mx::StreamOrDevice s) {
  return mx::random::uniform(low, high,
                             std::move(shape.value_or(std::vector<int>())),
                             dtype.value_or(mx::float32), std::move(key), s);
}

mx::array Normal(std::optional<std::vector<int>> shape,
                 std::optional<mx::Dtype> dtype,
                 std::optional<float> loc,
                 std::optional<float> scale,
                 std::optional<mx::array> key,
                 mx::StreamOrDevice s) {
  return mx::random::normal(
      std::move(shape.value_or(std::vector<int>())),
      dtype.value_or(mx::float32), loc.value_or(0), scale.value_or(1),
      std::move(key), s);
}

mx::array MultivariateNormal(const mx::array& mean,
                             const mx::array& conv,
                             std::optional<std::vector<int>> shape,
                             std::optional<mx::Dtype> dtype,
                             std::optional<mx::array> key,
                             mx::StreamOrDevice s) {
  return mx::random::multivariate_normal(
      mean, conv, std::move(shape.value_or(std::vector<int>())),
      dtype.value_or(mx::float32), std::move(key), s);
}

mx::array RandInt(const mx::array& low,
                  const mx::array& high,
                  std::optional<std::vector<int>> shape,
                  std::optional<mx::Dtype> dtype,
                  std::optional<mx::array> key,
                  mx::StreamOrDevice s) {
  return mx::random::randint(
      low, high, std::move(shape.value_or(std::vector<int>())),
      dtype.value_or(mx::int32), std::move(key), s);
}

mx::array Bernoulli(std::optional<mx::array> param,
                    std::optional<std::vector<int>> shape,
                    std::optional<mx::array> key,
                    mx::StreamOrDevice s) {
  mx::array p = std::move(param.value_or(mx::array(0.5)));
  if (shape)
    return mx::random::bernoulli(p, *shape, std::move(key), s);
  else
    return mx::random::bernoulli(p, std::move(key), s);
}

mx::array TruncatedNormal(const mx::array& lower,
                          const mx::array& upper,
                          std::optional<std::vector<int>> shape,
                          std::optional<mx::Dtype> dtype,
                          std::optional<mx::array> key,
                          mx::StreamOrDevice s) {
  if (shape) {
    return mx::random::truncated_normal(
        lower, upper, *shape, dtype.value_or(mx::float32), std::move(key), s);
  } else {
    return mx::random::truncated_normal(
        lower, upper, dtype.value_or(mx::float32), std::move(key), s);
  }
}

mx::array Gumbel(std::optional<std::vector<int>> shape,
                 std::optional<mx::Dtype> dtype,
                 std::optional<mx::array> key,
                 mx::StreamOrDevice s) {
  return mx::random::gumbel(std::move(shape.value_or(std::vector<int>())),
                            dtype.value_or(mx::float32), std::move(key), s);
}

mx::array Categorical(
    const mx::array& logits,
    std::optional<int> optional_axis,
    // Use variant to explicitly allow passing null/undefined as shape.
    std::optional<std::variant<std::monostate, std::vector<int>>> shape,
    std::optional<int> num_samples,
    std::optional<mx::array> key,
    mx::StreamOrDevice s) {
  bool has_shape = shape && std::get_if<std::vector<int>>(&shape.value());
  if (has_shape && num_samples) {
    throw std::invalid_argument(
        "[categorical] At most one of shape or num_samples can be specified.");
  }
  int axis = optional_axis.value_or(-1);
  if (has_shape) {
    return mx::random::categorical(logits, axis,
                                   std::get<std::vector<int>>(*shape), key, s);
  }
  if (num_samples)
    return mx::random::categorical(logits, axis, *num_samples, key, s);
  return mx::random::categorical(logits, axis, key, s);
}

mx::array Laplace(std::optional<std::vector<int>> shape,
                  std::optional<mx::Dtype> dtype,
                  std::optional<float> loc,
                  std::optional<float> scale,
                  std::optional<mx::array> key,
                  mx::StreamOrDevice s) {
  return mx::random::normal(
      std::move(shape.value_or(std::vector<int>())),
      dtype.value_or(mx::float32), loc.value_or(0), scale.value_or(1),
      std::move(key), s);
}

mx::array Permuation(std::variant<int, mx::array> x,
                     int axis,
                     std::optional<mx::array> key,
                     mx::StreamOrDevice s) {
  if (auto i = std::get_if<int>(&x); i) {
    return mx::random::permutation(*i, std::move(key), s);
  } else {
    return mx::random::permutation(std::move(std::get<mx::array>(x)),
                                   axis, std::move(key), s);
  }
}

}  // namespace random_ops

void InitRandom(napi_env env, napi_value exports) {
  napi_value random = ki::CreateObject(env);
  ki::Set(env, exports, "random", random);

  ki::Set(env, random,
          "seed", &mx::random::seed,
          "key", &mx::random::key,
          "split", &random_ops::Split,
          "uniform", &random_ops::Uniform,
          "normal", &random_ops::Normal,
          "multivariateNormal", &random_ops::MultivariateNormal,
          "randint", &random_ops::RandInt,
          "bernoulli", &random_ops::Bernoulli,
          "truncatedNormal", &random_ops::TruncatedNormal,
          "gumbel", &random_ops::Gumbel,
          "categorical", &random_ops::Categorical,
          "laplace", &random_ops::Laplace,
          "permuation", &random_ops::Permuation);
}
