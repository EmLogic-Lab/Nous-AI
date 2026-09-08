"""Load nous_mini.pt and generate text. Run this later on your PC too."""

from __future__ import annotations

import argparse

import torch

from train_nous_mini import Config, NousMini, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="nous_mini.pt")
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in Config.__dataclass_fields__})
    chars = ckpt["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = NousMini(cfg, vocab_size=len(chars)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = args.prompt if args.prompt else chars[0]
    unknown = [c for c in prompt if c not in stoi]
    if unknown:
        raise SystemExit(f"Prompt has characters the model never saw: {unknown!r}")

    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
