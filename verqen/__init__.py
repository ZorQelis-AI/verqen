"""VerQen by ZorQelis AI: an open decision model trained from scratch."""

from verqen.config import DataConfig, LoggingConfig, LossConfig, ModelConfig, TrainingConfig, VerqenConfig
from verqen.model.attention_mask import build_attention_mask, build_verqen_attention_mask
from verqen.model.verqen import DecisionOutput, VerqenModel
from verqen.tokenizer import VerqenTokenizer

__version__ = "1.0.0"


def __getattr__(name):
    # Lazy import: the inference API pulls in calibration/eval code.
    if name == "Verqen":
        from verqen.inference.api import Verqen
        return Verqen
    raise AttributeError(name)


__all__ = [
    "Verqen", "VerqenModel", "DecisionOutput", "VerqenTokenizer",
    "VerqenConfig", "ModelConfig", "TrainingConfig", "LossConfig", "DataConfig", "LoggingConfig",
    "build_attention_mask", "build_verqen_attention_mask", "__version__",
]
