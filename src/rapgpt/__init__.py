"""RapGPT: A Conditional Transformer for Rap Lyrics and Flow Generation."""

import os
import tempfile

# Ensure torch cache directories avoid permission issues on custom systems
if "TORCHINDUCTOR_CACHE_DIR" not in os.environ or ".unsloth" in os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "torchinductor_rapgpt")
if "TORCH_HOME" not in os.environ or ".unsloth" in os.environ.get("TORCH_HOME", ""):
    os.environ["TORCH_HOME"] = os.path.join(tempfile.gettempdir(), "torch_rapgpt")

from rapgpt.config import GenerationConfig, ModelConfig, TrainingConfig
from rapgpt.generator import RapGenerator
from rapgpt.metrics import analyze_rap_lyrics
from rapgpt.model import NanoGPT
from rapgpt.tokenizer import RapTokenizer

__version__ = "1.0.0"
__all__ = [
    "NanoGPT",
    "RapTokenizer",
    "RapGenerator",
    "ModelConfig",
    "GenerationConfig",
    "TrainingConfig",
    "analyze_rap_lyrics",
]
