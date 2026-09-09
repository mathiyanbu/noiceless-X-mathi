#include <gtest/gtest.h>
#include <concepts>
#include <span>
#include <vector>

// Verify C++20 standard support
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
T add_values(T a, T b) {
    return a + b;
}

TEST(SanityTest, BasicAssertion) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
}

TEST(SanityTest, Cpp20ConceptsAndSpan) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::span<float> view(data);
    
    EXPECT_EQ(view.size(), 4UL);
    EXPECT_FLOAT_EQ(view[0], 1.0f);
    EXPECT_FLOAT_EQ(add_values(view[0], view[1]), 3.0f);
}
