mod data;
mod lora;
mod model;
mod train;

use anyhow::Result;
use candle_core::Device;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Step 3: download BERT and verify it loads
    let loaded = model::load_bert("bert-base-uncased", &device)?;

    // Confirm the tokenizer file landed on disk
    println!("Tokenizer path: {}", loaded.tokenizer_path.display());

    Ok(())
}
