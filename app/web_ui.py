import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr
from rapgpt.config import GenerationConfig
from rapgpt.generator import RapGenerator
from rapgpt.metrics import analyze_rap_lyrics
from rapgpt.model import NanoGPT
from rapgpt.tokenizer import CADENCES, GENRES, MOODS, RHYME_SCHEMES, RapTokenizer

MODEL_PATH = os.getenv("RAPGPT_MODEL_PATH", "models/best_model.pt")
TOKENIZER_PATH = os.getenv("RAPGPT_TOKENIZER_PATH", "models/tokenizer.json")

generator = None


def get_generator():
    global generator
    if generator is None:
        if not Path(MODEL_PATH).exists() or not Path(TOKENIZER_PATH).exists():
            return None
        tok = RapTokenizer.load(TOKENIZER_PATH)
        model, _ = NanoGPT.load_pretrained(MODEL_PATH)
        generator = RapGenerator(model, tok)
    return generator


def generate_rap_lyrics(
    starter,
    genre,
    mood,
    cadence,
    rhyme,
    artist,
    song,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
):
    gen = get_generator()
    if gen is None:
        return (
            "Model checkpoint or tokenizer not found in models/",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        )

    prompt = gen.tokenizer.format_rap_prompt(
        genre=genre,
        mood=mood,
        cadence=cadence,
        rhyme=rhyme,
        artist=artist,
        song=song,
        starter=starter,
    )

    cfg = GenerationConfig(
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        strategy="top_p",
    )

    lyrics, stats = gen.generate(prompt, config=cfg, clean_output=True)
    analysis = analyze_rap_lyrics(lyrics)

    stat_str = f"{stats['tokens_generated']} tokens, {stats['elapsed_seconds']}s ({stats['tokens_per_second']} tok/s)"
    rhyme_scheme = analysis.get("rhyme_scheme", "N/A")
    rhyme_density = f"{analysis.get('rhyme_density', 0)}%"
    avg_syllables = str(analysis.get("avg_syllables_per_line", "N/A"))
    flow_rating = analysis.get("flow_rating", "N/A")

    return lyrics, stat_str, rhyme_scheme, rhyme_density, avg_syllables, flow_rating


def generate_poetry(prompt, max_tokens, temperature, top_p, rep_pen):
    gen = get_generator()
    if gen is None:
        return "Model not loaded.", "N/A"

    cfg = GenerationConfig(
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(rep_pen),
        strategy="top_p",
    )
    raw_prompt = f"<|start|>\n{prompt.strip()}"
    text, stats = gen.generate(raw_prompt, config=cfg, clean_output=False)
    cleaned = text.replace("<|start|>", "").replace("<|end|>", "").strip()
    stat_str = f"{stats['tokens_generated']} tokens ({stats['tokens_per_second']} tok/s)"
    return cleaned, stat_str


def analyze_text(custom_text):
    if not custom_text.strip():
        return "No text provided.", "", "", ""
    res = analyze_rap_lyrics(custom_text)
    return (
        res.get("rhyme_scheme", "N/A"),
        f"{res.get('rhyme_density', 0)}%",
        f"{res.get('avg_syllables_per_line')} (std: {res.get('syllable_std_dev')})",
        res.get("flow_rating", "N/A"),
    )


with gr.Blocks(title="RapGPT") as demo:
    gr.Markdown("# RapGPT")

    with gr.Tabs():
        with gr.Tab("Rap Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    starter_input = gr.Textbox(
                        label="Starter line",
                        placeholder="e.g. Started from the ground",
                        lines=2,
                    )
                    with gr.Row():
                        artist_input = gr.Textbox(label="Artist", placeholder="e.g. Tupac")
                        song_input = gr.Textbox(label="Song", placeholder="e.g. Changes")

                    with gr.Row():
                        genre_drop = gr.Dropdown(label="Genre", choices=GENRES, value="trap")
                        mood_drop = gr.Dropdown(label="Mood", choices=MOODS, value="hype")

                    with gr.Row():
                        cadence_drop = gr.Dropdown(label="Cadence", choices=CADENCES, value="bouncy")
                        rhyme_drop = gr.Dropdown(label="Rhyme Scheme", choices=RHYME_SCHEMES, value="multisyllabic")

                    with gr.Accordion("Generation parameters", open=False):
                        max_tokens_slider = gr.Slider(minimum=20, maximum=256, value=120, step=10, label="Max tokens")
                        temp_slider = gr.Slider(minimum=0.2, maximum=1.5, value=0.8, step=0.05, label="Temperature")
                        top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")
                        top_k_slider = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-k")
                        rep_pen_slider = gr.Slider(minimum=1.0, maximum=1.5, value=1.18, step=0.02, label="Repetition penalty")

                    gen_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1):
                    lyrics_output = gr.Textbox(label="Generated Lyrics", lines=14)
                    speed_box = gr.Label(label="Runtime")

                    with gr.Row():
                        metric_scheme = gr.Textbox(label="Rhyme Scheme")
                        metric_density = gr.Textbox(label="Rhyme Density")
                    with gr.Row():
                        metric_syllables = gr.Textbox(label="Avg Syllables/Line")
                        metric_flow = gr.Textbox(label="Flow")

            gen_btn.click(
                fn=generate_rap_lyrics,
                inputs=[
                    starter_input,
                    genre_drop,
                    mood_drop,
                    cadence_drop,
                    rhyme_drop,
                    artist_input,
                    song_input,
                    max_tokens_slider,
                    temp_slider,
                    top_p_slider,
                    top_k_slider,
                    rep_pen_slider,
                ],
                outputs=[
                    lyrics_output,
                    speed_box,
                    metric_scheme,
                    metric_density,
                    metric_syllables,
                    metric_flow,
                ],
            )

        with gr.Tab("Poetry Generation"):
            with gr.Row():
                with gr.Column():
                    poem_prompt = gr.Textbox(label="Prompt", value="The last light faded", lines=2)
                    poem_tokens = gr.Slider(minimum=20, maximum=256, value=100, step=10, label="Max tokens")
                    poem_temp = gr.Slider(minimum=0.2, maximum=1.5, value=0.7, step=0.05, label="Temperature")
                    poem_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")
                    poem_rep = gr.Slider(minimum=1.0, maximum=1.5, value=1.15, step=0.02, label="Repetition penalty")
                    poem_btn = gr.Button("Generate Poem", variant="primary")
                with gr.Column():
                    poem_output = gr.Textbox(label="Output", lines=12)
                    poem_stats = gr.Label(label="Runtime")

            poem_btn.click(
                fn=generate_poetry,
                inputs=[poem_prompt, poem_tokens, poem_temp, poem_top_p, poem_rep],
                outputs=[poem_output, poem_stats],
            )

        with gr.Tab("Linguistic Analysis"):
            custom_input = gr.Textbox(label="Lyrics text", placeholder="Paste lyrics here", lines=8)
            analyze_btn = gr.Button("Analyze")
            with gr.Row():
                an_scheme = gr.Textbox(label="Rhyme Scheme")
                an_density = gr.Textbox(label="Rhyme Density")
                an_syllables = gr.Textbox(label="Avg Syllables/Line")
                an_flow = gr.Textbox(label="Flow")

            analyze_btn.click(
                fn=analyze_text,
                inputs=[custom_input],
                outputs=[an_scheme, an_density, an_syllables, an_flow],
            )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
