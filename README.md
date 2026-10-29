# RapGPT: Conditional NanoGPT for Rap Generation

RapGPT is an autoregressive decoder-only Transformer (~23M parameters) for conditional generation of rap lyrics and poetry. It supports conditioning on genre, mood, cadence, and rhyme scheme, alongside automated flow and rhyme scheme metrics.

---

## Model Specifications

| Parameter | Value |
| :--- | :--- |
| Parameters | 23,033,856 |
| Layers (`n_layer`) | 6 |
| Heads (`n_head`) | 6 |
| Embedding Dim (`n_embed`) | 384 |
| Block Size (`block_size`) | 256 |
| FFN Multiplier | 4 |
| Vocab Size | 16,000 |
| Pre-training Corpus | 19,655 cleaned poems (`all_poems.txt`) |
| Domain Fine-tuning Corpus | 2,196 rap songs (`Raps.csv`) |

---

## Project Structure

```
RapGPT/
├── src/
│   └── rapgpt/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── tokenizer.py
│       ├── dataset.py
│       ├── generator.py
│       ├── trainer.py
│       ├── metrics.py
│       └── api.py
├── app/
│   └── web_ui.py
├── models/
│   ├── best_model.pt
│   └── tokenizer.json
├── data/
│   ├── raw/Raps.csv
│   └── processed/rap_training.txt
├── configs/
│   ├── model.yaml
│   └── finetune.yaml
├── tests/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_metrics.py
├── cli.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Model Info
```bash
python cli.py info
```

### 2. Generate Rap Lyrics
```bash
python cli.py generate --genre trap --mood hype --cadence bouncy --rhyme multisyllabic --prompt "Started from the ground" --analyze
```

### 3. Generate Poetry
```bash
python cli.py poetry --prompt "The last light faded" --max-tokens 80
```

### 4. Interactive Web UI
```bash
python cli.py ui --port 7860
```

### 5. REST API Server
```bash
python cli.py serve --port 8000
```

### 6. Training / Fine-Tuning
```bash
python cli.py train --data data/processed/rap_training.txt --steps 500 --batch-size 8 --lr 3e-5
```

### 7. Run Tests
```bash
pytest -v tests
```

---

## Supported Control Tokens

* **Genres**: `trap`, `boom_bap`, `conscious_rap`, `drill`, `grime`, `melodic_rap`, `alternative_hiphop`, `underground`, `experimental`, `mainstream`
* **Moods**: `confident`, `hype`, `introspective`, `aggressive`, `melancholic`, `rebellious`, `dark`, `playful`, `storytelling`, `motivational`
* **Cadences**: `bouncy`, `fast`, `relaxed`, `aggressive`, `triplet`, `syncopated`, `steady`, `melodic`
* **Rhyme Schemes**: `multisyllabic`, `internal`, `couplets`, `cross_rhyme`, `complex`, `aabb`, `abab`
