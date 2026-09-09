#pragma once

#include <vector>
#include <cstddef>
#include <span>

namespace noiselessx {
namespace dsp {

/**
 * @brief Normalized Least Mean Squares (NLMS) Adaptive Filter.
 * 
 * Implements exact dual-microphone acoustic noise cancellation:
 *   n_hat[n] = w^T[n] r[n]
 *   e[n]     = x[n] - n_hat[n]
 *   w[n+1]   = w[n] + (mu * e[n] * r[n]) / (epsilon + ||r[n]||^2)
 * 
 * Where:
 *   x[n] is the Primary (headphone) mic sample (speech + noise)
 *   r[n] is the Reference (error) mic sample (ambient noise)
 *   e[n] is the error output (cleaned speech)
 */
class NLMSFilter {
public:
    NLMSFilter() = default;
    explicit NLMSFilter(int filter_length, float learning_rate = 0.05f, float epsilon = 1e-6f);

    void initialize(int filter_length, float learning_rate, float epsilon);
    float process(float primary, float reference);
    void process_block(const float* primary, const float* reference, float* output_error, size_t count);
    void reset();

    [[nodiscard]] const std::vector<float>& get_weights() const noexcept { return weights; }
    [[nodiscard]] const std::vector<float>& get_history() const noexcept { return history; }
    [[nodiscard]] int get_filter_length() const noexcept { return filter_length; }
    [[nodiscard]] float get_learning_rate() const noexcept { return learning_rate; }
    [[nodiscard]] float get_epsilon() const noexcept { return epsilon; }

    void set_learning_rate(float mu) noexcept { learning_rate = mu; }
    void set_epsilon(float eps) noexcept { epsilon = eps; }

private:
    std::vector<float> weights;
    std::vector<float> history;

    int filter_length{256};
    float learning_rate{0.05f};
    float epsilon{1e-6f};
    size_t history_idx{0};
    bool is_initialized{false};
};

} // namespace dsp
} // namespace noiselessx
