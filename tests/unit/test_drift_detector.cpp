#include <gtest/gtest.h>
#include "embedded/audio/drift_detector.hpp"

using namespace noiselessx::audio;

TEST(DriftDetectorTest, ZeroDriftWhenSynchronized) {
    DriftDetector detector(16000, 10.0); // 16kHz, max 10ms

    // Both streams receive 160 samples concurrently
    detector.record_primary(160, 1000000);
    detector.record_reference(160, 1000000);

    EXPECT_EQ(detector.get_drift_samples(), 0);
    EXPECT_DOUBLE_EQ(detector.get_drift_ms(), 0.0);
    EXPECT_DOUBLE_EQ(detector.get_timestamp_drift_ms(), 0.0);
    EXPECT_FALSE(detector.is_warning_active());
}

TEST(DriftDetectorTest, PositiveAndNegativeSampleDrift) {
    DriftDetector detector(16000, 10.0);

    // Primary leads by 160 samples (10ms at 16kHz)
    detector.record_primary(320);
    detector.record_reference(160);

    EXPECT_EQ(detector.get_drift_samples(), 160);
    EXPECT_DOUBLE_EQ(detector.get_drift_ms(), 10.0); // 160 / 16000 * 1000 = 10ms
    EXPECT_FALSE(detector.is_warning_active());       // exactly at 10.0, not exceeded

    // Add 16 more primary samples -> 176 drift (11ms) -> exceeds threshold
    detector.record_primary(16);
    EXPECT_EQ(detector.get_drift_samples(), 176);
    EXPECT_DOUBLE_EQ(detector.get_drift_ms(), 11.0);
    EXPECT_TRUE(detector.is_warning_active());

    // Reference catches up and surpasses Primary
    detector.record_reference(352); // Ref total: 512, Prim total: 336 -> drift: -176
    EXPECT_EQ(detector.get_drift_samples(), -176);
    EXPECT_DOUBLE_EQ(detector.get_drift_ms(), -11.0);
    EXPECT_TRUE(detector.is_warning_active());
}

TEST(DriftDetectorTest, TimestampDriftCalculation) {
    DriftDetector detector(16000, 5.0);

    // Primary timestamp at 10 ms (10,000,000 ns)
    // Reference timestamp at 8 ms (8,000,000 ns)
    detector.record_primary(160, 10000000);
    detector.record_reference(160, 8000000);

    // Timestamp difference: 10ms - 8ms = 2.0ms
    EXPECT_DOUBLE_EQ(detector.get_timestamp_drift_ms(), 2.0);
}

TEST(DriftDetectorTest, ResetFunctionality) {
    DriftDetector detector(16000, 5.0);

    detector.record_primary(1600); // 100ms
    detector.record_reference(0);
    EXPECT_TRUE(detector.is_warning_active());

    detector.reset();
    EXPECT_EQ(detector.get_drift_samples(), 0);
    EXPECT_DOUBLE_EQ(detector.get_drift_ms(), 0.0);
    EXPECT_FALSE(detector.is_warning_active());
    EXPECT_EQ(detector.get_primary_total_frames(), 0UL);
    EXPECT_EQ(detector.get_reference_total_frames(), 0UL);
}
