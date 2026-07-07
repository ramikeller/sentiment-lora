# sentiment-lora

Binary sentiment classification using LoRA (Low-Rank Adaptation), written in Rust with the [candle](https://github.com/huggingface/candle) framework. We use a fine-tuned BERT model. 

## What it does

Takes a pre-trained `bert-base-uncased` model from Hugging Face and fine-tunes it on a labeled sentiment dataset (positive / negative) without modifying the original weights. Instead of updating all ~110M BERT parameters, LoRA inserts small trainable rank-decomposition matrices alongside the frozen attention weights — typically less than 0.1% of the total parameter count.

## Architecture

Current (classification head only):
```
Input text
    └─▶ WordPiece tokenizer (bert-base-uncased vocabulary)
            └─▶ BERT encoder (12 layers, frozen)
                    └─▶ [CLS] token hidden state (768-dim)
                            └─▶ Classification head (trainable) → positive / negative
```

Target (with LoRA):
```
Input text
    └─▶ WordPiece tokenizer (bert-base-uncased vocabulary)
            └─▶ BERT encoder (12 layers, frozen)
                    └─▶ [CLS] token hidden state (768-dim)
                            └─▶ LoRA-adapted attention layers (trainable)
                                    └─▶ Classification head → positive / negative
```

## Project structure

```
src/
  data.rs    — CSV loading, tokenization, padding
  model.rs   — BERT loading, SentimentModel (BERT + classification head)
  lora.rs    — LoRA A/B matrix pairs (in progress)
  train.rs   — training loop and evaluation
  main.rs    — entrypoint
data/
  sentiment.csv          — labeled training examples
classifier.safetensors   — saved classification head weights (produced by training)
```

## Dependencies

- [candle](https://github.com/huggingface/candle) — Hugging Face's ML framework for Rust
- [hf-hub](https://github.com/huggingface/hf-hub) — downloads model weights from Hugging Face
- [tokenizers](https://github.com/huggingface/tokenizers) — fast WordPiece tokenization

## Running

**Train the classification head** (saves weights to `classifier.safetensors` when done):

```bash
cargo run --release
```

**Run inference on a sentence** (loads saved weights automatically):

```bash
cargo run --release -- "This movie was absolutely fantastic!"
# → positive
```

On first run, `bert-base-uncased` weights (~440 MB) are downloaded and cached in `~/.cache/huggingface/hub/`. Subsequent runs use the cache.

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project skeleton | done |
| 2 | Data loading & tokenization | done |
| 3 | BERT model loading | done |
| 4 | Classification head | done |
| 5 | Training loop & evaluation | done |
| 6 | Command-line inference | done |
| 7 | LoRA layers | in progress |
