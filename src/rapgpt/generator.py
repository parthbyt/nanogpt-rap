import time
from typing import Generator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from rapgpt.config import GenerationConfig
from rapgpt.model import NanoGPT
from rapgpt.tokenizer import RapTokenizer


class RapGenerator:
    def __init__(
        self,
        model: NanoGPT,
        tokenizer: RapTokenizer,
        default_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = default_config or GenerationConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        clean_output: bool = True,
    ) -> Tuple[str, dict]:
        cfg = config or self.config

        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [self.tokenizer.token_to_id("<|start|>") or 3]

        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_token_ids = []

        stop_token_ids = set()
        for s in cfg.stop_tokens:
            tid = self.tokenizer.token_to_id(s)
            if tid is not None:
                stop_token_ids.add(tid)

        start_time = time.time()

        for step in range(cfg.max_new_tokens):
            idx_cond = idx[:, -self.model.block_size :]

            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :].clone()

            if idx.size(1) > 0 and cfg.repetition_penalty != 1.0:
                for tid in set(idx[0].tolist()):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= cfg.repetition_penalty
                    else:
                        logits[0, tid] *= cfg.repetition_penalty

            logits = logits / max(cfg.temperature, 1e-8)

            if cfg.strategy == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            elif cfg.strategy == "top_k":
                v, _ = torch.topk(logits, min(cfg.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            elif cfg.strategy == "top_p":
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                remove = cumulative_probs > cfg.top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                probs = F.softmax(sorted_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(sorted_idx, 1, sampled)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if token_id in stop_token_ids and step > 10:
                break

            generated_token_ids.append(token_id)
            idx = torch.cat([idx, next_token], dim=1)

        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_token_ids) / max(elapsed, 1e-4)

        raw_output = self.tokenizer.decode(input_ids + generated_token_ids, skip_special_tokens=False)
        output_text = self.tokenizer.clean_lyrics(raw_output) if clean_output else raw_output

        stats = {
            "tokens_generated": len(generated_token_ids),
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "prompt_tokens": len(input_ids),
            "total_tokens": len(input_ids) + len(generated_token_ids),
        }

        return output_text, stats
