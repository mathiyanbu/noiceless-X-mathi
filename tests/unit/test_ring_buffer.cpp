#include <gtest/gtest.h>
#include "embedded/audio/ring_buffer.hpp"
#include <vector>
#include <numeric>

using namespace noiselessx::audio;

TEST(RingBufferTest, BasicPushPop) {
    SpscRingBuffer<int, 8> rb;
    EXPECT_TRUE(rb.empty());
    EXPECT_FALSE(rb.full());
    EXPECT_EQ(rb.size(), 0UL);
    EXPECT_EQ(rb.capacity(), 8UL);

    EXPECT_TRUE(rb.push(42));
    EXPECT_EQ(rb.size(), 1UL);
    EXPECT_FALSE(rb.empty());

    int val = 0;
    EXPECT_TRUE(rb.pop(val));
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(rb.empty());
}

TEST(RingBufferTest, FifoOrder) {
    SpscRingBuffer<int, 16> rb;
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(rb.push(i * 10));
    }
    EXPECT_EQ(rb.size(), 10UL);

    for (int i = 0; i < 10; ++i) {
        int val = -1;
        EXPECT_TRUE(rb.pop(val));
        EXPECT_EQ(val, i * 10);
    }
    EXPECT_TRUE(rb.empty());
}

TEST(RingBufferTest, WraparoundIndexing) {
    SpscRingBuffer<int, 4> rb; // Capacity 4
    
    // Cycle 100 times through a small buffer to verify pointer masking & wraparound
    for (int cycle = 0; cycle < 100; ++cycle) {
        EXPECT_TRUE(rb.push(cycle));
        EXPECT_TRUE(rb.push(cycle + 1000));

        int val1 = 0, val2 = 0;
        EXPECT_TRUE(rb.pop(val1));
        EXPECT_TRUE(rb.pop(val2));

        EXPECT_EQ(val1, cycle);
        EXPECT_EQ(val2, cycle + 1000);
    }
    EXPECT_TRUE(rb.empty());
    EXPECT_EQ(rb.overflow_count(), 0UL);
}

TEST(RingBufferTest, OverflowHandling) {
    SpscRingBuffer<int, 4> rb; // Capacity 4

    // Fill buffer to capacity
    EXPECT_TRUE(rb.push(1));
    EXPECT_TRUE(rb.push(2));
    EXPECT_TRUE(rb.push(3));
    EXPECT_TRUE(rb.push(4));
    EXPECT_TRUE(rb.full());

    // Pushing 5th item should fail and record overflow
    EXPECT_FALSE(rb.push(5));
    EXPECT_FALSE(rb.push(6));
    EXPECT_EQ(rb.overflow_count(), 2UL);

    // Existing items should be intact
    int val = 0;
    EXPECT_TRUE(rb.pop(val));
    EXPECT_EQ(val, 1);

    // Now one slot is free
    EXPECT_TRUE(rb.push(7));
    EXPECT_EQ(rb.overflow_count(), 2UL);
}

TEST(RingBufferTest, BatchOperations) {
    SpscRingBuffer<int, 16> rb;

    std::vector<int> input(8);
    std::iota(input.begin(), input.end(), 100);

    size_t pushed = rb.push_batch(input.data(), input.size());
    EXPECT_EQ(pushed, 8UL);
    EXPECT_EQ(rb.size(), 8UL);

    std::vector<int> output(8, 0);
    size_t popped = rb.pop_batch(output.data(), 8);
    EXPECT_EQ(popped, 8UL);
    EXPECT_EQ(output, input);
    EXPECT_TRUE(rb.empty());
}

TEST(RingBufferTest, BatchOverflow) {
    SpscRingBuffer<int, 8> rb; // Capacity 8

    std::vector<int> input(12, 5); // 12 elements into capacity 8
    size_t pushed = rb.push_batch(input.data(), input.size());

    EXPECT_EQ(pushed, 8UL);
    EXPECT_EQ(rb.overflow_count(), 4UL);
    EXPECT_TRUE(rb.full());
}
