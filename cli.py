import argparse
import os
import sys
import tempfile
from pathlib import Path

if "TORCHINDUCTOR_CACHE_DIR" not in os.environ or ".unsloth" in os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "torchinductor_rapgpt")
if "TORCH_HOME" not in os.environ or ".unsloth" in os.environ.get("TORCH_HOME", ""):
    os.environ["TORCH_HOME"] = os.path.join(tempfile.gettempdir(), "torch_rapgpt")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rapgpt.config import GenerationConfig, ModelConfig, TrainingConfig
from rapgpt.generator import RapGenerator
from rapgpt.metrics import analyze_rap_lyrics
from rapgpt.model import NanoGPT
from rapgpt.tokenizer import RapTokenizer


def get_default_paths():
    finetuned = "models/rap_finetuned.pt"
    pretrained = "models/best_model.pt"
    default_model = finetuned if Path(finetuned).exists() else pretrained
    model_path = os.getenv("RAPGPT_MODEL_PATH", default_model)
    tokenizer_path = os.getenv("RAPGPT_TOKENIZER_PATH", "models/tokenizer.json")
    return model_path, tokenizer_path


def cmd_generate(args):
    model_path, tok_path = get_default_paths()
    if args.checkpoint:
        model_path = args.checkpoint

    tok = RapTokenizer.load(tok_path)
    model, _ = NanoGPT.load_pretrained(model_path)
    generator = RapGenerator(model, tok)

    prompt = tok.format_rap_prompt(
        genre=args.genre,
        mood=args.mood,
        cadence=args.cadence,
        rhyme=args.rhyme,
        artist=args.artist,
        song=args.song,
        starter=args.prompt,
    )

    cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.rep_penalty,
        strategy=args.strategy,
    )

    lyrics, stats = generator.generate(prompt, config=cfg, clean_output=True)
    print("--- Prompt ---")
    print(prompt.strip())
    print("\n--- Output ---")
    print(lyrics)
    print(f"\n({stats['tokens_generated']} tokens in {stats['elapsed_seconds']}s, {stats['tokens_per_second']} tok/s)")

    if args.analyze and lyrics:
        analysis = analyze_rap_lyrics(lyrics)
        print("\n--- Analysis ---")
        print(f"Rhyme Scheme : {analysis.get('rhyme_scheme')}")
        print(f"Rhyme Density: {analysis.get('rhyme_density')}%")
        print(f"Avg Syllables: {analysis.get('avg_syllables_per_line')}")
        print(f"Flow         : {analysis.get('flow_rating')}")


def cmd_poetry(args):
    model_path = args.checkpoint or "models/best_model.pt"
    tok_path = os.getenv("RAPGPT_TOKENIZER_PATH", "models/tokenizer.json")
    tok = RapTokenizer.load(tok_path)
    model, _ = NanoGPT.load_pretrained(model_path)
    generator = RapGenerator(model, tok)

    raw_prompt = f"<|start|>\n{args.prompt.strip()}"
    cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
    )

    text, stats = generator.generate(raw_prompt, config=cfg, clean_output=False)
    cleaned = text.replace("<|start|>", "").replace("<|end|>", "").strip()
    print("--- Poetry Output ---")
    print(cleaned)


def cmd_train(args):
    from rapgpt.dataset import create_dataloaders
    from rapgpt.trainer import RapTrainer

    model_path, tok_path = get_default_paths()
    tok = RapTokenizer.load(tok_path)

    train_loader, val_loader = create_dataloaders(
        text_path=args.data,
        tokenizer=tok,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )

    if args.from_scratch:
        model = NanoGPT(ModelConfig(block_size=args.block_size))
    else:
        model, _ = NanoGPT.load_pretrained(model_path)

    cfg = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_checkpoint_name=args.output,
    )

    trainer = RapTrainer(model, train_loader, val_loader, config=cfg)
    print(f"Training for {args.steps} steps...")
    res = trainer.train(max_steps=args.steps)
    print(f"Done. Best Val Loss: {res['best_val_loss']:.4f}")


def cmd_analyze(args):
    text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    if not text:
        print("Error: Provide either --text or --file")
        return

    analysis = analyze_rap_lyrics(text)
    for k, v in analysis.items():
        print(f"{k:22s}: {v}")


def cmd_ui(args):
    print(f"Starting Web UI on http://127.0.0.1:{args.port} ...", flush=True)
    from app.web_ui import demo
    demo.launch(server_port=args.port, server_name="127.0.0.1", share=args.share)


def cmd_serve(args):
    import uvicorn
    uvicorn.run("rapgpt.api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_info(args):
    model_path, tok_path = get_default_paths()
    if Path(model_path).exists():
        model, meta = NanoGPT.load_pretrained(model_path)
        params = model.count_parameters()
        print(f"Checkpoint Path : {model_path}")
        print(f"Step            : {meta.get('step')}")
        print(f"Val Loss        : {meta.get('val_loss')}")
        print(f"Parameters      : {params:,}")
        print(f"Block Size      : {model.block_size}")

    if Path(tok_path).exists():
        tok = RapTokenizer.load(tok_path)
        print(f"Tokenizer Path  : {tok_path}")
        print(f"Vocab Size      : {tok.vocab_size:,}")


def main():
    parser = argparse.ArgumentParser(description="RapGPT CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate conditional rap lyrics")
    gen_parser.add_argument("--prompt", "-p", type=str, default="Look at the sky")
    gen_parser.add_argument("--genre", "-g", type=str, default="trap")
    gen_parser.add_argument("--mood", "-m", type=str, default="hype")
    gen_parser.add_argument("--cadence", "-c", type=str, default="bouncy")
    gen_parser.add_argument("--rhyme", "-r", type=str, default="multisyllabic")
    gen_parser.add_argument("--artist", "-a", type=str, default=None)
    gen_parser.add_argument("--song", type=str, default=None)
    gen_parser.add_argument("--max-tokens", "-n", type=int, default=120)
    gen_parser.add_argument("--temperature", "-t", type=float, default=0.8)
    gen_parser.add_argument("--top-p", type=float, default=0.9)
    gen_parser.add_argument("--top-k", type=int, default=40)
    gen_parser.add_argument("--rep-penalty", type=float, default=1.18)
    gen_parser.add_argument("--strategy", type=str, default="top_p", choices=["top_p", "top_k", "greedy"])
    gen_parser.add_argument("--checkpoint", type=str, default=None)
    gen_parser.add_argument("--analyze", action="store_true")
    gen_parser.set_defaults(func=cmd_generate)

    poem_parser = subparsers.add_parser("poetry", help="Generate poem verse")
    poem_parser.add_argument("--prompt", "-p", type=str, default="The last light faded")
    poem_parser.add_argument("--max-tokens", "-n", type=int, default=100)
    poem_parser.add_argument("--temperature", "-t", type=float, default=0.7)
    poem_parser.add_argument("--top-p", type=float, default=0.9)
    poem_parser.add_argument("--rep-penalty", type=float, default=1.15)
    poem_parser.add_argument("--checkpoint", type=str, default=None)
    poem_parser.set_defaults(func=cmd_poetry)

    train_parser = subparsers.add_parser("train", help="Train or fine-tune model")
    train_parser.add_argument("--data", "-d", type=str, default="data/processed/rap_training.txt")
    train_parser.add_argument("--steps", "-s", type=int, default=100)
    train_parser.add_argument("--batch-size", "-b", type=int, default=8)
    train_parser.add_argument("--block-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=3e-5)
    train_parser.add_argument("--eval-interval", type=int, default=50)
    train_parser.add_argument("--eval-iters", type=int, default=5)
    train_parser.add_argument("--output", "-o", type=str, default="rap_finetuned.pt")
    train_parser.add_argument("--from-scratch", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    an_parser = subparsers.add_parser("analyze", help="Analyze text metrics")
    an_parser.add_argument("--text", "-t", type=str, default=None)
    an_parser.add_argument("--file", "-f", type=str, default=None)
    an_parser.set_defaults(func=cmd_analyze)

    ui_parser = subparsers.add_parser("ui", help="Launch web UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true")
    ui_parser.set_defaults(func=cmd_ui)

    serve_parser = subparsers.add_parser("serve", help="Launch API server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    info_parser = subparsers.add_parser("info", help="Display diagnostics")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
