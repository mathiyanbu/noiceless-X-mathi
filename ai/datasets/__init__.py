"""
AI Datasets Subsystem: Research dataset acquisition, indexing, and manifests.
"""

from ai.datasets.sources import (
    CLEAN_SPEECH_DATASETS,
    NOISE_DATASETS,
    RIR_DATASETS,
    IMPULSIVE_CLASSES,
    STEADY_STATE_NOISE_CLASSES,
    get_all_registered_noise_classes,
    is_impulsive_class
)

__all__ = [
    "CLEAN_SPEECH_DATASETS",
    "NOISE_DATASETS",
    "RIR_DATASETS",
    "IMPULSIVE_CLASSES",
    "STEADY_STATE_NOISE_CLASSES",
    "get_all_registered_noise_classes",
    "is_impulsive_class"
]
