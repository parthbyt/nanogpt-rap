"""Unit tests for NanoGPT architecture and checkpoint compatibility."""

import os
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rapgpt.config import ModelConfig
from rapgpt.model import NanoGPT


def test_model_initialization():
    cfg = ModelConfig(vocab_size=1000, block_size=64, n_embed=64, n_head=2, n_layer=2)
    model = NanoGPT(cfg)
    assert model.block_size == 64
    assert len(model.blocks) == 2


def test_model_forward_shape():
    cfg = ModelConfig(vocab_size=500, block_size=32, n_embed=64, n_head=2, n_layer=2)
    model = NanoGPT(cfg)
    idx = torch.randint(0, 500, (2, 16))
    logits, loss = model(idx)

    assert logits.shape == (2, 16, 500)
    assert loss is None


def test_model_loss_computation():
    cfg = ModelConfig(vocab_size=500, block_size=32, n_embed=64, n_head=2, n_layer=2)
    model = NanoGPT(cfg)
    idx = torch.randint(0, 500, (2, 16))
    targets = torch.randint(0, 500, (2, 16))
    logits, loss = model(idx, targets=targets)

    assert loss is not None
    assert loss.item() > 0.0


def test_checkpoint_compatibility():
    ckpt_path = Path("models/best_model.pt")
    if not ckpt_path.is_file():
        pytest.skip("models/best_model.pt not present")

    model, meta = NanoGPT.load_pretrained(ckpt_path)
    assert model.count_parameters() == 23033856
    assert meta["step"] == 35000
    assert abs(meta["val_loss"] - 4.446) < 0.01
