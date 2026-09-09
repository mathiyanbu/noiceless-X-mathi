"""
Pydantic Schemas for NOICELESSX API Endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' if both backend & runtime are active; 'degraded' if runtime is offline")
    backend: str = "healthy"
    runtime_connected: bool
    runtime_error: Optional[str] = None
    runtime_state: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class CpuMetrics(BaseModel):
    overall_pct: float
    cores_pct: List[float]


class MemoryMetrics(BaseModel):
    total_mb: float
    used_mb: float
    free_mb: float


class SystemResponse(BaseModel):
    cpu: CpuMetrics
    temperature_c: float
    temperature_available: bool
    memory: MemoryMetrics
    source: str = "kernel"
    timestamp: float = Field(default_factory=time.time)


class AudioDeviceItem(BaseModel):
    card_index: int
    device_index: int
    device_type: str
    identifier: str
    name: str


class AudioConfigInfo(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 10
    hop_ms: int = 5
    primary_device: str = "hw:CARD=Headset,DEV=0"
    reference_device: str = "hw:CARD=ErrorMic,DEV=0"
    output_device: str = "hw:CARD=Headset,DEV=0"
    period_size: int = 160
    buffer_size: int = 640


class AudioDevicesResponse(BaseModel):
    config: AudioConfigInfo
    devices: List[AudioDeviceItem]


class ModelResponse(BaseModel):
    model_path: str
    model_name: str
    quantization: str
    execution_provider: str
    intra_op_threads: int
    input_shape: List[str]
    output_shape: List[str]


class StageLatencies(BaseModel):
    capture_us: float = 0.0
    preprocessing_us: float = 0.0
    stft_us: float = 0.0
    ai_inference_us: float = 0.0
    istft_us: float = 0.0
    nlms_us: float = 0.0
    fusion_us: float = 0.0
    playback_queue_us: float = 0.0
    total_processing_us: float = 0.0
    end_to_end_latency_ms: float = 0.0


class MetricsResponse(BaseModel):
    latencies: StageLatencies
    rtf: float
    processed_frames: int = 0
    dropped_frames: int = 0
    alsa_xruns: Dict[str, int] = Field(default_factory=lambda: {"primary": 0, "reference": 0, "playback": 0})
    drift_ms: float = 0.0
    drift_samples: int = 0
    drift_warning: bool = False
    fusion_mode: str = "NORMAL"
    current_lambda: float = 0.5
    impulse_envelope_gain: float = 1.0
    ai_confidence: float = 1.0
    impulse_probability: float = 0.0
    vad_probability: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class BypassRequest(BaseModel):
    bypass: Optional[bool] = None


class RuntimeCommandResponse(BaseModel):
    status: str
    command: str
    message: str
    runtime_state: Optional[str] = None
