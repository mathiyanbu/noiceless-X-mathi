#include "nlms.hpp"
#include <cmath>
#include <algorithm>

namespace noiselessx {
namespace dsp {

NLMSFilter::NLMSFilter(int filter_length, float learning_rate, float epsilon) {
    initialize(filter_length, learning_rate, epsilon);
}

void NLMSFilter::initialize(int filter_length_in, float learning_rate_in, float epsilon_in) {
    filter_length = std::max(1, filter_length_in);
    learning_rate = learning_rate_in;
    epsilon = std::max(1e-12f, epsilon_in);

    weights.assign(static_cast<size_t>(filter_length), 0.0f);
    history.assign(static_cast<size_t>(filter_length), 0.0f);
    history_idx = 0;
    is_initialized = true;
}

void NLMSFilter::reset() {
    std::fill(weights.begin(), weights.end(), 0.0f);
    std::fill(history.begin(), history.end(), 0.0f);
    history_idx = 0;
}

float NLMSFilter::process(float primary, float reference) {
    if (!is_initialized || filter_length <= 0) {
        return primary;
    }

    // 1. Insert current reference sample r[n] at head of circular history
    history[history_idx] = reference;

    // 2. Compute estimated noise n_hat[n] = w^T[n] r[n] and norm ||r[n]||^2
    float n_hat = 0.0f;
    float norm_sq = 0.0f;

    const size_t L = static_cast<size_t>(filter_length);
    const float* w_ptr = weights.data();
    const float* h_ptr = history.data();

    // Split circular index loop into two contiguous linear spans for SIMD auto-vectorization
    // Part 1: i from 0 to history_idx -> delay index is history_idx - i
    const size_t part1_len = history_idx + 1;
    for (size_t i = 0; i < part1_len; ++i) {
        const float r_i = h_ptr[history_idx - i];
        n_hat += w_ptr[i] * r_i;
        norm_sq += r_i * r_i;
    }

    // Part 2: i from history_idx + 1 to L - 1 -> delay index is history_idx - i + L
    for (size_t i = part1_len; i < L; ++i) {
        const float r_i = h_ptr[history_idx + L - i];
        n_hat += w_ptr[i] * r_i;
        norm_sq += r_i * r_i;
    }

    // 3. Compute error e[n] = x[n] - n_hat[n]
    const float e = primary - n_hat;

    // 4. Update equation: w[n+1] = w[n] + (mu * e[n] * r[n]) / (epsilon + ||r[n]||^2)
    const float step = (learning_rate * e) / (epsilon + norm_sq);

    // Apply weight updates across the two contiguous spans
    float* w_mut = weights.data();
    for (size_t i = 0; i < part1_len; ++i) {
        w_mut[i] += step * h_ptr[history_idx - i];
    }
    for (size_t i = part1_len; i < L; ++i) {
        w_mut[i] += step * h_ptr[history_idx + L - i];
    }

    // 5. Advance circular pointer for next sample
    history_idx = (history_idx + 1) % L;

    return e;
}

void NLMSFilter::process_block(const float* primary, const float* reference, float* output_error, size_t count) {
    if (!primary || !output_error || count == 0) return;

    if (!reference || !is_initialized) {
        std::copy_n(primary, count, output_error);
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        output_error[i] = process(primary[i], reference[i]);
    }
}

} // namespace dsp
} // namespace noiselessx
