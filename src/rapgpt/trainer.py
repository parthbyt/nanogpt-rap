import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

# Environment safeguards for custom platforms
if "TORCHINDUCTOR_CACHE_DIR" not in os.environ or ".unsloth" in os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "torchinductor_rapgpt")
if "TORCH_HOME" not in os.environ or ".unsloth" in os.environ.get("TORCH_HOME", ""):
    os.environ["TORCH_HOME"] = os.path.join(tempfile.gettempdir(), "torch_rapgpt")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from rapgpt.config import TrainingConfig
from rapgpt.model import NanoGPT


class RapTrainer:
    def __init__(
        self,
        model: NanoGPT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.95))

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=max(1, self.config.warmup_steps),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.max_steps - self.config.warmup_steps),
            eta_min=self.config.min_learning_rate,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps],
        )

        self.best_val_loss = float("inf")
        self.current_step = 0

    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        iters = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                iters += 1
                if iters >= self.config.eval_iters:
                    break

        self.model.train()
        return total_loss / max(1, iters)

    def train(
        self,
        max_steps: Optional[int] = None,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> dict:
        steps_to_run = max_steps or self.config.max_steps
        self.model.train()

        train_iter = iter(self.train_loader)
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = checkpoint_dir / self.config.save_checkpoint_name

        history = []
        for step in range(1, steps_to_run + 1):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(x, targets=y)
            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            self.current_step += 1

            train_loss = loss.item()

            if step % self.config.eval_interval == 0 or step == steps_to_run:
                val_loss = self.evaluate()
                lr = self.optimizer.param_groups[0]["lr"]
                history.append({
                    "step": step,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "lr": lr,
                })

                if callback:
                    callback(step, train_loss, val_loss)
                else:
                    print(
                        f"step={step:5d} | "
                        f"train={train_loss:.4f} | "
                        f"val={val_loss:.4f} | "
                        f"lr={lr:.2e}"
                    )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(save_path, val_loss)
                    print(f"Saved checkpoint (val={val_loss:.4f}) to {save_path}")

        return {
            "total_steps": steps_to_run,
            "best_val_loss": self.best_val_loss,
            "save_path": str(save_path),
            "history": history,
        }

    def save_checkpoint(self, path: Union[str, Path], val_loss: float):
        torch.save(
            {
                "step": self.current_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            str(path),
        )
