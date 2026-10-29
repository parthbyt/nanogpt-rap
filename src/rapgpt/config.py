from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 256
    n_embed: int = 384
    n_head: int = 6
    n_layer: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1


@dataclass
class GenerationConfig:
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.18
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    strategy: str = "top_p"
    stop_tokens: List[str] = field(
        default_factory=lambda: ["<|end|>", "<|endoftext|>", "<|endlyrics|>"]
    )


@dataclass
class TrainingConfig:
    batch_size: int = 8
    block_size: int = 256
    learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.1
    max_steps: int = 2500
    warmup_steps: int = 100
    eval_interval: int = 100
    eval_iters: int = 20
    grad_clip: float = 1.0
    device: Optional[str] = None
    checkpoint_dir: str = "models"
    save_checkpoint_name: str = "rap_finetuned.pt"
