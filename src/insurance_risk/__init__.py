"""Insurance risk analytics package."""

from .data import (
    DEFAULT_TARGET,
    REAL_DATA_PATH,
    SyntheticDataConfig,
    dataset_mode,
    generate_demo_dataset,
    load_dataset,
)
from .modeling import evaluate_model, load_model_artifact, predict_premium, train_model

__all__ = [
    "DEFAULT_TARGET",
    "REAL_DATA_PATH",
    "SyntheticDataConfig",
    "dataset_mode",
    "generate_demo_dataset",
    "load_dataset",
    "evaluate_model",
    "load_model_artifact",
    "predict_premium",
    "train_model",
]
