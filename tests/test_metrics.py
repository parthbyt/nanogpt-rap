"""Unit tests for rhyme analysis and syllable counting."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rapgpt.metrics import analyze_rap_lyrics, count_syllables, detect_rhyme_scheme


def test_syllable_counter():
    assert count_syllables("rap") == 1
    assert count_syllables("microphone") == 3
    assert count_syllables("transformer") == 3
    assert count_syllables("rhythm") >= 1


def test_rhyme_scheme_aabb():
    lines = [
        "I walk down the street today",
        "Nothing can stand in my way",
        "Looking high up at the moon",
        "I will be returning soon",
    ]
    scheme, density = detect_rhyme_scheme(lines)
    assert scheme == "AABB"
    assert density == 1.0


def test_analyze_rap_lyrics():
    text = "We started in the dark tonight\nWe shining under golden light\nMoving steady with the crew\nGot nothing else to prove to you"
    res = analyze_rap_lyrics(text)
    assert res["line_count"] == 4
    assert res["word_count"] > 10
    assert "rhyme_scheme" in res
    assert "avg_syllables_per_line" in res
