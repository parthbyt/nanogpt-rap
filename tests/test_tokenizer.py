"""Unit tests for RapTokenizer and prompt formatting."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rapgpt.tokenizer import RapTokenizer, SPECIAL_TOKENS


def test_tokenizer_special_tokens():
    tok_path = Path("models/tokenizer.json")
    if not tok_path.is_file():
        pytest.skip("models/tokenizer.json not present")

    tok = RapTokenizer.load(tok_path)
    assert tok.vocab_size == 16000

    for st in SPECIAL_TOKENS:
        tid = tok.token_to_id(st)
        assert tid is not None, f"Special token {st} missing from vocab"


def test_tokenizer_roundtrip():
    tok_path = Path("models/tokenizer.json")
    if not tok_path.is_file():
        pytest.skip("models/tokenizer.json not present")

    tok = RapTokenizer.load(tok_path)
    text = "Spitting fire on the microphone every night"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded.strip() == text.strip()


def test_format_rap_prompt():
    prompt = RapTokenizer.format_rap_prompt(
        genre="drill",
        mood="aggressive",
        cadence="fast",
        rhyme="internal",
        artist="Pop Smoke",
        starter="Step in the room",
    )
    assert "<|genre|>drill" in prompt
    assert "<|mood|>aggressive" in prompt
    assert "<|artist|>Pop Smoke" in prompt
    assert "<|lyrics|>\nStep in the room" in prompt


def test_clean_lyrics():
    raw = "<|genre|>trap\n<|lyrics|>\nLine one\nLine two\n<|endlyrics|>\n<|end|>"
    cleaned = RapTokenizer.clean_lyrics(raw)
    assert cleaned == "Line one\nLine two"
