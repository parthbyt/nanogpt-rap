import re
import statistics
from typing import Dict, List, Optional, Tuple


def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 1

    if len(word) <= 3:
        return 1
    if word.endswith("es") or word.endswith("ed"):
        word = word[:-2]
    elif word.endswith("e") and not word.endswith("le"):
        word = word[:-1]

    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    return max(1, count)


def extract_rhyme_ending(word: str) -> str:
    word = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not word:
        return ""
    vowels = "aeiouy"
    last_vowel_idx = -1
    for i in range(len(word) - 1, -1, -1):
        if word[i] in vowels:
            last_vowel_idx = i
            if i > 0 and word[i - 1] in vowels:
                last_vowel_idx = i - 1
            break

    if last_vowel_idx != -1:
        return word[last_vowel_idx:]
    return word[-3:] if len(word) >= 3 else word


def detect_rhyme_scheme(lines: List[str]) -> Tuple[str, float]:
    end_words = []
    for line in lines:
        words = re.findall(r"\b[a-zA-Z']+\b", line)
        if words:
            end_words.append(words[-1])

    if len(end_words) < 2:
        return "Unknown", 0.0

    suffixes = [extract_rhyme_ending(w) for w in end_words]
    scheme_labels = []
    label_map = {}
    current_char_code = ord("A")

    for suffix in suffixes:
        matched = False
        for key in label_map:
            if suffix and key and (suffix == key or suffix.endswith(key) or key.endswith(suffix)):
                scheme_labels.append(label_map[key])
                matched = True
                break
        if not matched:
            char = chr(current_char_code)
            label_map[suffix] = char
            scheme_labels.append(char)
            current_char_code += 1

    scheme_str = "".join(scheme_labels)

    counts = {}
    for char in scheme_labels:
        counts[char] = counts.get(char, 0) + 1

    rhymed_lines = sum(c for c in counts.values() if c > 1)
    density = rhymed_lines / len(scheme_labels) if scheme_labels else 0.0

    return scheme_str, round(density, 3)


def analyze_rap_lyrics(text: str) -> Dict[str, object]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return {"error": "Empty text"}

    all_words = re.findall(r"\b[a-zA-Z']+\b", text)
    total_words = len(all_words)
    unique_words = len(set(w.lower() for w in all_words))
    ttr = round(unique_words / max(total_words, 1), 3)

    syllable_counts = []
    for line in lines:
        words = re.findall(r"\b[a-zA-Z']+\b", line)
        s_count = sum(count_syllables(w) for w in words)
        syllable_counts.append(s_count)

    avg_syllables = round(statistics.mean(syllable_counts), 1) if syllable_counts else 0
    std_syllables = (
        round(statistics.stdev(syllable_counts), 2)
        if len(syllable_counts) > 1
        else 0.0
    )

    scheme_str, rhyme_density = detect_rhyme_scheme(lines)

    if std_syllables <= 2.0:
        flow_rating = "Steady"
    elif std_syllables <= 3.5:
        flow_rating = "Standard"
    else:
        flow_rating = "Dynamic"

    return {
        "line_count": len(lines),
        "word_count": total_words,
        "unique_words": unique_words,
        "ttr": ttr,
        "avg_syllables_per_line": avg_syllables,
        "syllable_std_dev": std_syllables,
        "flow_rating": flow_rating,
        "rhyme_scheme": scheme_str[:16],
        "rhyme_density": round(rhyme_density * 100, 1),
    }
