# Nous-Mini from-scratch training

Trains a tiny GPT from random weights on public Tiny Shakespeare text.

This is practice language learning, not a personal companion yet.
Do not upload private chats, diaries, or school files.

## Run on Google Colab

1. Open https://colab.research.google.com
2. File → New notebook
3. Runtime → Change runtime type → GPU (T4) → Save
4. Upload `train_nous_mini.py` or paste its contents into a cell
5. Run:

```python
!python train_nous_mini.py
```

6. When it finishes, download `nous_mini.pt` from the file sidebar

On a free Colab GPU, 3000 steps usually takes a few minutes.

## What good looks like

- Before training: random characters
- During training: train/val loss should fall
- After training: broken Shakespeare-like English, not clean chat yet

That means the model learned patterns from scratch.

## Next

After this works, we change the dataset to short User/Nous dialogues
and connect `nous_mini.pt` to your FastAPI `/api/chat` route.
