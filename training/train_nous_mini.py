"""
Nous-Mini — train a tiny GPT from scratch.

This starts from RANDOM weights. It does not download a pretrained brain.
Intended for Google Colab (GPU) or any machine with PyTorch.

Public practice data only. Do not paste private chats or diaries here.
"""

from __future__ import annotations

import math
import os
import time
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# CONFIG
# =========================================

@dataclass
class Config:
    # Tiny Shakespeare is public-domain practice text (~1 MB).
    data_url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt"
    )
    data_path: str = "input.txt"
    ckpt_path: str = "nous_mini.pt"
    # If this file exists, continue from that brain instead of random weights.
    resume_path: str = "nous_mini_resume.pt"

    # Small on purpose. Quality comes later by growing these numbers.
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    batch_size: int = 32
    max_steps: int = 1500
    eval_every: int = 250
    eval_batches: int = 20
    learning_rate: float = 3e-3
    resume_learning_rate: float = 8e-4
    weight_decay: float = 0.1

    generate_tokens: int = 400
    seed: int = 42


# =========================================
# DATA
# =========================================

def download_data(cfg: Config) -> str:
    if not os.path.exists(cfg.data_path):
        print(f"Downloading practice data to {cfg.data_path} ...")
        urllib.request.urlretrieve(cfg.data_url, cfg.data_path)
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {cfg.data_path}")
    return text


def build_tokenizer(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    return chars, encode, decode


def make_splits(data: torch.Tensor):
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(split: torch.Tensor, cfg: Config, device: torch.device):
    ix = torch.randint(len(split) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([split[i : i + cfg.block_size] for i in ix])
    y = torch.stack([split[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# =========================================
# MODEL
# =========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NousMini(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError("Sequence longer than block_size")
        pos = torch.arange(0, t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.9):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def estimate_loss(model: NousMini, splits, cfg: Config, device: torch.device) -> dict:
    model.eval()
    out = {}
    for name, split in splits.items():
        losses = []
        for _ in range(cfg.eval_batches):
            x, y = get_batch(split, cfg, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"Device: {device}")

    text = download_data(cfg)

    resumed = False
    resume_ckpt = None
    if os.path.exists(cfg.resume_path):
        resume_ckpt = torch.load(cfg.resume_path, map_location=device, weights_only=False)
        chars = resume_ckpt["chars"]
        extra = sorted(set(text) - set(chars))
        if extra:
            raise SystemExit(
                "New text has characters the old model never saw: "
                + repr("".join(extra))
                + ". Add those letters to a from-scratch run, or remove them from the text."
            )
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(s: str) -> list[int]:
            return [stoi[c] for c in s]

        def decode(ids: list[int]) -> str:
            return "".join(itos[i] for i in ids)

        resumed = True
        print(f"Resuming from {cfg.resume_path}")
    else:
        chars, encode, decode = build_tokenizer(text)
        print("No resume file found. Starting from random weights.")

    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = make_splits(data)
    splits = {"train": train_data, "val": val_data}

    model = NousMini(cfg, vocab_size).to(device)
    if resumed:
        model.load_state_dict(resume_ckpt["model"])
        print(f"Loaded previous val_loss={resume_ckpt.get('val_loss', 'unknown')}")

    print(f"Parameters: {count_params(model):,}")

    lr = cfg.resume_learning_rate if resumed else cfg.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg.weight_decay
    )
    print(f"Learning rate: {lr}")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    before = decode(model.generate(start, 200, temperature=1.0)[0].tolist())
    print("\n===== BEFORE THIS RUN =====\n")
    print(before)
    print("\n===== TRAINING =====\n")

    model.train()
    t0 = time.time()
    best_val = math.inf

    for step in range(1, cfg.max_steps + 1):
        x, y = get_batch(train_data, cfg, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, splits, cfg, device)
            dt = time.time() - t0
            print(
                f"step {step:5d}/{cfg.max_steps} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"{dt:.1f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "chars": chars,
                    "val_loss": best_val,
                    "step": step,
                }
                torch.save(ckpt, cfg.ckpt_path)
                print(f"  saved {cfg.ckpt_path} (val {best_val:.4f})")

    after = decode(model.generate(start, cfg.generate_tokens, temperature=0.9)[0].tolist())
    print("\n===== AFTER TRAINING =====\n")
    print(after)
    print(f"\nCheckpoint: {os.path.abspath(cfg.ckpt_path)}")
    print("Download that file. It is your trained Nous-Mini.")


if __name__ == "__main__":
    main()
