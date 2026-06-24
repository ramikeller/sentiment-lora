mod data;
mod lora;
mod model;
mod train;

use anyhow::Result;
use candle_core::Device;
use std::path::Path;

fn main() -> Result<()> {
    let device = Device::Cpu;

    let loaded = model::load_bert("bert-base-uncased", &device)?;
    let tokenizer_path = loaded.tokenizer_path.clone();
    let model = model::SentimentModel::new(loaded)?;

    let tokenizer = data::load_tokenizer(&tokenizer_path)?;
    let samples = data::load_dataset(Path::new("data/sentiment.csv"), &tokenizer, 128)?;
    println!("Loaded {} samples", samples.len());

    let baseline = train::evaluate(&model, &samples, &device)?;
    println!("Accuracy (before training): {:.1}%", baseline * 100.0);

    train::train(&model, &samples, &device, 10, 1e-3)?;

    Ok(())
}
