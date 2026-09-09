#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <array>
#include <algorithm>
#include <span>
#include <type_traits>

namespace noiselessx::audio {

/**
 * @brief Lock-free, bounded Single-Producer Single-Consumer (SPSC) Ring Buffer.
 * 
 * Designed for real-time audio threads:
 * - Zero heap allocation during push/pop.
 * - Lock-free and wait-free.
 * - Power-of-two capacity for fast bitwise masking.
 * - Tracks buffer overflows without blocking or throwing.
 */
template <typename T, size_t Capacity = 1024>
class SpscRingBuffer {
    static_assert((Capacity != 0) && ((Capacity & (Capacity - 1)) == 0),
                  "SpscRingBuffer Capacity must be a non-zero power of 2");

public:
    static constexpr size_t BufferCapacity = Capacity;
    static constexpr size_t Mask = Capacity - 1;

    SpscRingBuffer() : write_idx_(0), read_idx_(0), overflow_count_(0) {}

    // Non-copyable, non-movable to guarantee thread-safe memory addresses
    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;
    SpscRingBuffer(SpscRingBuffer&&) = delete;
    SpscRingBuffer& operator=(SpscRingBuffer&&) = delete;

    /**
     * @brief Push a single element into the buffer (Producer only).
     * @return true if pushed, false if buffer is full (overflow recorded).
     */
    bool push(const T& item) {
        const size_t current_read = read_idx_.load(std::memory_order_acquire);
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);

        if (current_write - current_read >= Capacity) {
            overflow_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        buffer_[current_write & Mask] = item;
        write_idx_.store(current_write + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Push a single element by moving it into the buffer (Producer only).
     */
    bool push(T&& item) {
        const size_t current_read = read_idx_.load(std::memory_order_acquire);
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);

        if (current_write - current_read >= Capacity) {
            overflow_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        buffer_[current_write & Mask] = std::move(item);
        write_idx_.store(current_write + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pop a single element from the buffer (Consumer only).
     * @return true if popped, false if buffer is empty.
     */
    bool pop(T& item) {
        const size_t current_write = write_idx_.load(std::memory_order_acquire);
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);

        if (current_read == current_write) {
            return false;
        }

        item = std::move(buffer_[current_read & Mask]);
        read_idx_.store(current_read + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Push a contiguous batch of items into the buffer (Producer only).
     * @return Number of items successfully pushed.
     */
    size_t push_batch(const T* src, size_t count) {
        if (!src || count == 0) return 0;

        const size_t current_read = read_idx_.load(std::memory_order_acquire);
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        const size_t occupied = current_write - current_read;
        const size_t available = (occupied < Capacity) ? (Capacity - occupied) : 0;

        const size_t to_write = std::min(count, available);
        if (to_write < count) {
            overflow_count_.fetch_add(count - to_write, std::memory_order_relaxed);
        }

        for (size_t i = 0; i < to_write; ++i) {
            buffer_[(current_write + i) & Mask] = src[i];
        }

        if (to_write > 0) {
            write_idx_.store(current_write + to_write, std::memory_order_release);
        }

        return to_write;
    }

    /**
     * @brief Push a span of items into the buffer (Producer only).
     */
    size_t push_batch(std::span<const T> items) {
        return push_batch(items.data(), items.size());
    }

    /**
     * @brief Pop a batch of items from the buffer (Consumer only).
     * @return Number of items successfully popped.
     */
    size_t pop_batch(T* dst, size_t max_count) {
        if (!dst || max_count == 0) return 0;

        const size_t current_write = write_idx_.load(std::memory_order_acquire);
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);
        const size_t available = current_write - current_read;

        const size_t to_read = std::min(max_count, available);
        for (size_t i = 0; i < to_read; ++i) {
            dst[i] = std::move(buffer_[(current_read + i) & Mask]);
        }

        if (to_read > 0) {
            read_idx_.store(current_read + to_read, std::memory_order_release);
        }

        return to_read;
    }

    /**
     * @brief Pop a batch of items into a span (Consumer only).
     */
    size_t pop_batch(std::span<T> dst) {
        return pop_batch(dst.data(), dst.size());
    }

    /**
     * @brief Returns current number of unread items.
     */
    [[nodiscard]] size_t size() const noexcept {
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);
        return (current_write >= current_read) ? (current_write - current_read) : 0;
    }

    [[nodiscard]] static constexpr size_t capacity() noexcept {
        return Capacity;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] bool full() const noexcept {
        return size() >= Capacity;
    }

    [[nodiscard]] uint64_t overflow_count() const noexcept {
        return overflow_count_.load(std::memory_order_relaxed);
    }

    void reset_overflow_count() noexcept {
        overflow_count_.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Reset read and write pointers (Must NOT be called concurrently with push/pop).
     */
    void clear() noexcept {
        read_idx_.store(0, std::memory_order_relaxed);
        write_idx_.store(0, std::memory_order_relaxed);
        overflow_count_.store(0, std::memory_order_relaxed);
    }

private:
    // Separate cache lines to eliminate false sharing between producer and consumer
    alignas(64) std::atomic<size_t> write_idx_{0};
    alignas(64) std::atomic<size_t> read_idx_{0};
    alignas(64) std::atomic<uint64_t> overflow_count_{0};

    // Statically allocated internal buffer (zero dynamic allocation)
    alignas(64) std::array<T, Capacity> buffer_{};
};

} // namespace noiselessx::audio
