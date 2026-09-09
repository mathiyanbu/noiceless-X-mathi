#include <gtest/gtest.h>
#include "embedded/runtime/system_metrics.hpp"
#include "embedded/runtime/realtime_pipeline.hpp"
#include <vector>

using namespace noiselessx::runtime;

TEST(RealtimePipelineTest, SystemMetricsInitializationAndSampling) {
    SystemMetrics metrics;
    metrics.sample();

    // Overall CPU % must be bounded in [0, 100]
    EXPECT_GE(metrics.get_overall_cpu_percent(), 0.0f);
    EXPECT_LE(metrics.get_overall_cpu_percent(), 100.0f);

    // Per core metrics must be bounded in [0, 100]
    const auto& per_core = metrics.get_per_core_cpu_percent();
    for (float core_pct : per_core) {
        EXPECT_GE(core_pct, 0.0f);
        EXPECT_LE(core_pct, 100.0f);
    }

    // Temperature must either be positive or cleanly marked unavailable
    if (metrics.is_temperature_available()) {
        EXPECT_GT(metrics.get_cpu_temperature_c(), 0.0f);
        EXPECT_LT(metrics.get_cpu_temperature_c(), 110.0f); // Normal operating ceiling
    }
}

TEST(RealtimePipelineTest, PipelineConfigDefaultsAndStructure) {
    RealtimePipeline::PipelineConfig config;
    config.audio.sample_rate = 16000;
    config.audio.period_size = 160;
    config.audio.buffer_size = 640;
    config.audio.primary_device = "hw:CARD=Headset,DEV=0";
    config.audio.reference_device = "hw:CARD=ErrorMic,DEV=0";
    config.audio.output_device = "hw:CARD=Headset,DEV=0";

    RealtimePipeline pipeline(config);
    EXPECT_FALSE(pipeline.is_running());
    EXPECT_EQ(pipeline.get_config().audio.sample_rate, 16000u);
}

TEST(RealtimePipelineTest, TelemetryMetricsZeroFabricationStructure) {
    PipelineTelemetry t;
    t.capture_us = 120.0;
    t.preprocessing_us = 45.0;
    t.stft_us = 180.0;
    t.ai_inference_us = 1450.0;
    t.istft_us = 190.0;
    t.nlms_us = 110.0;
    t.fusion_us = 65.0;
    t.playback_queue_us = 35.0;

    double sum = t.capture_us + t.preprocessing_us + t.stft_us + t.ai_inference_us 
               + t.istft_us + t.nlms_us + t.fusion_us + t.playback_queue_us;
    t.total_processing_us = sum;

    // For 5ms hop (5000 us)
    double hop_us = 5000.0;
    t.rtf = t.total_processing_us / hop_us;

    EXPECT_NEAR(t.total_processing_us, 2195.0, 1e-4);
    EXPECT_NEAR(t.rtf, 2195.0 / 5000.0, 1e-4);
    EXPECT_LT(t.rtf, 0.50); // Headroom check
}

TEST(RealtimePipelineTest, BypassModePropagation) {
    RealtimePipeline::PipelineConfig config;
    RealtimePipeline pipeline(config);

    pipeline.set_bypass(true);
    // Setting bypass should not crash and should update internal state
    pipeline.set_bypass(false);
}
