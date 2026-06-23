mod data;
mod lora;
mod model;
mod train;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;

    let loaded = model::load_bert("bert-base-uncased", &device)?;
    let tokenizer_path = loaded.tokenizer_path.clone();
    let model = model::SentimentModel::new(loaded)?;

    // Smoke test: dummy batch [1, 128] to verify shapes are wired correctly.
    let input_ids = Tensor::zeros((1usize, 128usize), DType::U32, &device)?;
    let attention_mask = Tensor::ones((1usize, 128usize), DType::U32, &device)?;
    let logits = model.forward(&input_ids, &attention_mask)?;

    println!("Logits shape: {:?}", logits.shape());
    println!("Tokenizer:    {}", tokenizer_path.display());

    Ok(())
}
