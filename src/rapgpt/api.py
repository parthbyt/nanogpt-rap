import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rapgpt.config import GenerationConfig, ModelConfig
from rapgpt.generator import RapGenerator
from rapgpt.metrics import analyze_rap_lyrics
from rapgpt.model import NanoGPT
from rapgpt.tokenizer import CADENCES, GENRES, MOODS, RHYME_SCHEMES, RapTokenizer

_model: Optional[NanoGPT] = None
_tokenizer: Optional[RapTokenizer] = None
_generator: Optional[RapGenerator] = None


def get_generator() -> RapGenerator:
    global _model, _tokenizer, _generator
    if _generator is None:
        model_path = os.getenv("RAPGPT_MODEL_PATH", "models/best_model.pt")
        tok_path = os.getenv("RAPGPT_TOKENIZER_PATH", "models/tokenizer.json")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model checkpoint not found: {model_path}")
        if not os.path.exists(tok_path):
            raise RuntimeError(f"Tokenizer not found: {tok_path}")

        _tokenizer = RapTokenizer.load(tok_path)
        _model, _ = NanoGPT.load_pretrained(model_path)
        _generator = RapGenerator(_model, _tokenizer)
    return _generator


app = FastAPI(title="RapGPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    starter: Optional[str] = None
    genre: Optional[str] = "trap"
    mood: Optional[str] = "hype"
    cadence: Optional[str] = "bouncy"
    rhyme: Optional[str] = "multisyllabic"
    artist: Optional[str] = None
    song: Optional[str] = None

    max_tokens: int = Field(120, ge=10, le=512)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=100)
    repetition_penalty: float = Field(1.18, ge=1.0, le=2.0)


class AnalyzeRequest(BaseModel):
    lyrics: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "genres": GENRES,
        "moods": MOODS,
        "cadences": CADENCES,
        "rhyme_schemes": RHYME_SCHEMES,
        "parameters": 23033856,
        "layers": 6,
        "heads": 6,
        "n_embed": 384,
        "vocab_size": 16000,
        "block_size": 256,
    }


@app.post("/generate")
def generate_lyrics(req: GenerateRequest):
    try:
        gen = get_generator()
        prompt = gen.tokenizer.format_rap_prompt(
            genre=req.genre,
            mood=req.mood,
            cadence=req.cadence,
            rhyme=req.rhyme,
            artist=req.artist,
            song=req.song,
            starter=req.starter,
        )

        cfg = GenerationConfig(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            strategy="top_p",
        )

        lyrics, stats = gen.generate(prompt, config=cfg, clean_output=True)
        analysis = analyze_rap_lyrics(lyrics) if lyrics else {}

        return {
            "prompt": prompt,
            "lyrics": lyrics,
            "stats": stats,
            "analysis": analysis,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return analyze_rap_lyrics(req.lyrics)
