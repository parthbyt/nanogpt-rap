from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|unk|>",
    "<|start|>",
    "<|end|>",
    "<|artist|>",
    "<|song|>",
    "<|genre|>",
    "<|mood|>",
    "<|rhyme|>",
    "<|cadence|>",
    "<|lyrics|>",
    "<|endlyrics|>",
]

GENRES = [
    "mainstream",
    "conscious_rap",
    "trap",
    "boom_bap",
    "grime",
    "melodic_rap",
    "alternative_hiphop",
    "experimental",
    "underground",
    "drill",
]

MOODS = [
    "confident",
    "introspective",
    "aggressive",
    "melancholic",
    "rebellious",
    "dark",
    "hype",
    "romantic",
    "playful",
    "vulnerable",
    "storytelling",
    "motivational",
    "emotional",
]

CADENCES = [
    "bouncy",
    "fast",
    "relaxed",
    "aggressive",
    "triplet",
    "syncopated",
    "steady",
    "melodic",
]

RHYME_SCHEMES = [
    "multisyllabic",
    "internal",
    "couplets",
    "cross_rhyme",
    "complex",
    "aabb",
    "abab",
]


class RapTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.special_tokens = SPECIAL_TOKENS

    @classmethod
    def load(cls, path: Union[str, Path] = "models/tokenizer.json"):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def train(
        cls,
        corpus_path: Union[str, Path],
        output_path: Union[str, Path] = "models/tokenizer.json",
        vocab_size: int = 16000,
        min_frequency: int = 2,
    ):
        corpus_path = Path(corpus_path)
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )

        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenizer.train_from_iterator([text], trainer=trainer)
        tokenizer.decoder = ByteLevelDecoder()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))
        return cls(tokenizer)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tokenizer.id_to_token(token_id)

    @staticmethod
    def format_rap_prompt(
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        cadence: Optional[str] = None,
        rhyme: Optional[str] = None,
        artist: Optional[str] = None,
        song: Optional[str] = None,
        starter: Optional[str] = None,
    ) -> str:
        tags = []
        if artist:
            tags.append(f"<|artist|>{artist.strip()}")
        if song:
            tags.append(f"<|song|>{song.strip()}")
        if genre:
            tags.append(f"<|genre|>{genre.strip()}")
        if mood:
            tags.append(f"<|mood|>{mood.strip()}")
        if rhyme:
            tags.append(f"<|rhyme|>{rhyme.strip()}")
        if cadence:
            tags.append(f"<|cadence|>{cadence.strip()}")

        prefix = "\n".join(tags)
        if prefix:
            prefix += "\n\n"

        prefix += "<|lyrics|>\n"
        if starter and starter.strip():
            prefix += starter.strip()

        return prefix

    @staticmethod
    def clean_lyrics(text: str) -> str:
        if "<|lyrics|>" in text:
            text = text.split("<|lyrics|>")[-1]

        for token in SPECIAL_TOKENS:
            text = text.replace(token, "")

        lines = [line.strip() for line in text.split("\n")]
        cleaned = []
        consecutive_blanks = 0
        for line in lines:
            if not line:
                consecutive_blanks += 1
                if consecutive_blanks <= 1:
                    cleaned.append("")
            else:
                consecutive_blanks = 0
                cleaned.append(line)

        return "\n".join(cleaned).strip()
