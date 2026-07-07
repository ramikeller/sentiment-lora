use anyhow::{Context, Result};
use tokenizers::Tokenizer;
use std::path::Path;

// One tokenized training example.
// input_ids:      the token ID sequence, including [CLS] and [SEP], padded to max_len
// attention_mask: 1 for real tokens, 0 for padding — same length as input_ids
// label:          0 = negative, 1 = positive
pub struct SentimentSample {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub label: u32,
}

// Load a BERT tokenizer from a local tokenizer.json file.
// The tokenizer.json encodes the full vocabulary and merging rules —
// we'll download it from Hugging Face in main.rs (Step 3).
pub fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}

// Tokenize a single string into input_ids and attention_mask.
// max_len should be <= 512 for BERT (its hard architectural limit).
pub fn tokenize(text: &str, tokenizer: &Tokenizer, max_len: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    // encode() runs the full tokenization pipeline:
    //   raw text → WordPiece subwords → integer IDs
    // add_special_tokens=true prepends [CLS] (101) and appends [SEP] (102)
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

    // Truncate if the text is longer than max_len.
    // We keep the [SEP] token at the end by truncating from the middle.
    if input_ids.len() > max_len {
        input_ids.truncate(max_len - 1);
        input_ids.push(102); // [SEP]
        attention_mask.truncate(max_len);
    }

    // Pad shorter sequences with zeros up to max_len.
    // Padding ID 0 is conventional for BERT; the attention mask hides it.
    let pad_len = max_len.saturating_sub(input_ids.len());
    input_ids.extend(std::iter::repeat(0).take(pad_len));
    attention_mask.extend(std::iter::repeat(0).take(pad_len));

    Ok((input_ids, attention_mask))
}

// Read the CSV and return one SentimentSample per row.
// Expected CSV columns: text (string), label (0 or 1)
pub fn load_dataset(
    csv_path: &Path,
    tokenizer: &Tokenizer,
    max_len: usize,
) -> Result<Vec<SentimentSample>> {
    let mut reader = csv::Reader::from_path(csv_path)
        .with_context(|| format!("could not open {}", csv_path.display()))?;

    let mut samples = Vec::new();

    for (i, result) in reader.records().enumerate() {
        let record = result.with_context(|| format!("bad CSV row {i}"))?;

        let text = record.get(0).with_context(|| format!("missing text at row {i}"))?;
        let label: u32 = record
            .get(1)
            .with_context(|| format!("missing label at row {i}"))?
            .trim()
            .parse()
            .with_context(|| format!("label at row {i} is not 0 or 1"))?;

        let (input_ids, attention_mask) = tokenize(text, tokenizer, max_len)?;

        samples.push(SentimentSample {
            input_ids,
            attention_mask,
            label,
        });
    }

    Ok(samples)
}
