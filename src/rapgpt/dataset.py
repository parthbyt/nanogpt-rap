import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rapgpt.tokenizer import RapTokenizer

CHAR_MAP = {
    "\ufffd": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2014": "-",
    "\u2013": "-",
    "\u2026": "...",
    "\u00a0": " ",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for old, new in CHAR_MAP.items():
        text = text.replace(old, new)
    return text


def clean_formatting(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def remove_duplicate_stanzas(poem: str) -> str:
    seen = set()
    cleaned = []
    stanzas = [s.strip() for s in re.split(r"\n\s*\n", poem) if s.strip()]
    for stanza in stanzas:
        key = stanza.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(stanza)
    return "\n\n".join(cleaned)


def process_raps_csv(
    csv_path: Union[str, Path] = "data/raw/Raps.csv",
    output_txt: Union[str, Path] = "data/processed/rap_training.txt",
) -> int:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            lyrics = str(row.get("lyrics", "")).strip()
            if not lyrics:
                continue

            artist = str(row.get("artist", "")).strip()
            song = str(row.get("song", "")).strip()
            mood = str(row.get("mood", "")).strip()
            genre = str(row.get("genre", "")).strip()
            rhyme = str(row.get("rhyme_scheme", "")).strip()
            cadence = str(row.get("cadence", "")).strip()

            f.write("<|start|>")
            if artist:
                f.write(f"<|artist|>{artist}\n")
            if song:
                f.write(f"<|song|>{song}\n")
            if genre:
                f.write(f"<|genre|>{genre}\n")
            if mood:
                f.write(f"<|mood|>{mood}\n")
            if rhyme:
                f.write(f"<|rhyme|>{rhyme}\n")
            if cadence:
                f.write(f"<|cadence|>{cadence}\n\n")

            f.write("<|lyrics|>\n")
            f.write(lyrics)
            f.write("\n<|endlyrics|>\n\n")
            f.write("<|end|>\n\n")
            count += 1

    return count


class CausalLMDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int = 256):
        self.tokens = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_dataloaders(
    text_path: Union[str, Path],
    tokenizer: RapTokenizer,
    block_size: int = 256,
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    text_path = Path(text_path)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text)
    tensor_ids = torch.tensor(ids, dtype=torch.long)

    split_idx = int(len(tensor_ids) * (1.0 - val_split))
    train_tokens = tensor_ids[:split_idx]
    val_tokens = tensor_ids[split_idx:]

    train_dataset = CausalLMDataset(train_tokens, block_size=block_size)
    val_dataset = CausalLMDataset(val_tokens, block_size=block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
