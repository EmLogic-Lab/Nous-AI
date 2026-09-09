"""
Test a saved Nous-Mini checkpoint.

Upload this file plus nous_mini.pt (your v3 / latest) in Colab.
This loads the SAVED brain, not the leftover overfit model in memory.
"""

from __future__ import annotations

import torch

from train_nous_mini import Config, NousMini, pick_device


CKPT = "nous_mini.pt"
TOKENS = 180
TEMPERATURE = 0.7

TESTS = [
    "User: hi\nNous:",
    "User: hello\nNous:",
    "User: I am tired\nNous:",
    "User: I feel sad\nNous:",
    "User: thank you\nNous:",
    "User: who are you\nNous:",
    "User: I had a long day\nNous:",
    "User: I like blue cars\nNous:",
]


def main() -> None:
    device = pick_device()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in Config.__dataclass_fields__})
    chars = ckpt["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = NousMini(cfg, vocab_size=len(chars)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Device: {device}")
    print(f"Loaded {CKPT} | val_loss={ckpt.get('val_loss')} | step={ckpt.get('step')}")
    print()

    for prompt in TESTS:
        missing = [c for c in prompt if c not in stoi]
        if missing:
            print(f"SKIP {prompt!r} unknown chars {missing!r}")
            continue
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
        out = model.generate(idx, TOKENS, temperature=TEMPERATURE)[0].tolist()
        text = "".join(itos[i] for i in out)
        extra = text[len(prompt):]
        extra = extra.split("\nUser:")[0]
        print("=" * 40)
        print(prompt + extra.strip())
        print()


if __name__ == "__main__":
    main()
