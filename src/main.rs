mod data;
mod lora;
mod model;
mod train;

use anyhow::Result;
use candle_core::Device;
use std::path::Path;

const CHECKPOINT: &str = "classifier.safetensors";

fn main() -> Result<()> {
    let device = Device::Cpu;

    let loaded = model::load_bert("bert-base-uncased", &device)?;
    let tokenizer_path = loaded.tokenizer_path.clone();
    let mut model = model::SentimentModel::new(loaded)?;

    let tokenizer = data::load_tokenizer(&tokenizer_path)?;

    let args: Vec<String> = std::env::args().collect();
    if let Some(sentence) = args.get(1) {
        if Path::new(CHECKPOINT).exists() {
            model.var_map.load(CHECKPOINT)?;
        } else {
            eprintln!("Warning: no checkpoint found at '{CHECKPOINT}', using random weights. Run without arguments to train first.");
        }
        let (input_ids, attention_mask) = data::tokenize(sentence, &tokenizer, 128)?;
        let input_ids = candle_core::Tensor::from_vec(input_ids, (1, 128), &device)?;
        let attention_mask = candle_core::Tensor::from_vec(attention_mask, (1, 128), &device)?;
        let logits = model.forward(&input_ids, &attention_mask)?;
        let pred = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;
        println!("{}", if pred == 1 { "positive" } else { "negative" });
        return Ok(());
    }

    let samples = data::load_dataset(Path::new("data/sentiment.csv"), &tokenizer, 128)?;
    println!("Loaded {} samples", samples.len());

    let baseline = train::evaluate(&model, &samples, &device)?;
    println!("Accuracy (before training): {:.1}%", baseline * 100.0);

    train::train(&model, &samples, &device, 10, 1e-3)?;

    model.var_map.save(CHECKPOINT)?;
    println!("Classifier weights saved to '{CHECKPOINT}'");

    Ok(())
}
