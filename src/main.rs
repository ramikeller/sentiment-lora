mod data;
mod lora;
mod model;
mod train;

use anyhow::{Context, Result};
use candle_core::Device;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::seq::SliceRandom;
use std::path::{Path, PathBuf};

const CHECKPOINT: &str = "classifier.safetensors";
const MAX_TRAIN_SAMPLES: usize = 100;
const MAX_VAL_SAMPLES: usize = 50;

fn download_sst2() -> Result<(PathBuf, PathBuf)> {
    let api = Api::new().context("failed to create HF Hub client")?;
    let repo = api.repo(Repo::new("stanfordnlp/sst2".to_string(), RepoType::Dataset));
    println!("Fetching SST-2 dataset...");
    let train = repo
        .get("data/train-00000-of-00001.parquet")
        .context("failed to fetch SST-2 train split")?;
    let val = repo
        .get("data/validation-00000-of-00001.parquet")
        .context("failed to fetch SST-2 validation split")?;
    Ok((train, val))
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let loaded = model::load_bert("bert-base-uncased", &device)?;
    let tokenizer_path = loaded.tokenizer_path.clone();
    let mut model = model::SentimentModel::new(loaded)?;

    let tokenizer = data::load_tokenizer(&tokenizer_path)?;

    let args: Vec<String> = std::env::args().collect();
    let flag = args.get(1).map(|s| s.as_str());

    if flag == Some("--eval") {
        if Path::new(CHECKPOINT).exists() {
            model.var_map.load(CHECKPOINT)?;
        } else {
            eprintln!("Warning: no checkpoint found at '{CHECKPOINT}', evaluating with random weights.");
        }
        let mut rng = rand::thread_rng();
        let (_, val_path) = download_sst2()?;
        let mut samples = data::load_sst2(&val_path, &tokenizer, 128)?;
        samples.shuffle(&mut rng);
        samples.truncate(MAX_VAL_SAMPLES);
        println!("Loaded {} validation samples", samples.len());
        let accuracy = train::evaluate(&model, &samples, &device)?;
        println!("Accuracy: {:.1}%", accuracy * 100.0);
        return Ok(());
    }

    if flag == Some("--train") {
        let mut rng = rand::thread_rng();
        let (train_path, val_path) = download_sst2()?;
        let mut train_samples = data::load_sst2(&train_path, &tokenizer, 128)?;
        train_samples.shuffle(&mut rng);
        train_samples.truncate(MAX_TRAIN_SAMPLES);
        let mut val_samples = data::load_sst2(&val_path, &tokenizer, 128)?;
        val_samples.shuffle(&mut rng);
        val_samples.truncate(MAX_VAL_SAMPLES);
        println!("Loaded {} train / {} validation samples", train_samples.len(), val_samples.len());

        let baseline = train::evaluate(&model, &val_samples, &device)?;
        println!("Accuracy (before training): {:.1}%", baseline * 100.0);

        train::train(&model, &train_samples, &device, 3, 2e-4)?;

        let accuracy = train::evaluate(&model, &val_samples, &device)?;
        println!("Accuracy (after training): {:.1}%", accuracy * 100.0);

        model.var_map.save(CHECKPOINT)?;
        println!("Classifier weights saved to '{CHECKPOINT}'");
        return Ok(());
    }

    if Path::new(CHECKPOINT).exists() {
        model.var_map.load(CHECKPOINT)?;
    } else {
        eprintln!("Warning: no checkpoint found at '{CHECKPOINT}', using random weights. Run --train first.");
    }

    if let Some(sentence) = args.get(1) {
        let (input_ids, attention_mask) = data::tokenize(sentence, &tokenizer, 128)?;
        let input_ids = candle_core::Tensor::from_vec(input_ids, (1, 128), &device)?;
        let attention_mask = candle_core::Tensor::from_vec(attention_mask, (1, 128), &device)?;
        let logits = model.forward(&input_ids, &attention_mask)?;
        let pred = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;
        println!("{}", if pred == 1 { "positive" } else { "negative" });
        return Ok(());
    }

    eprintln!("Usage: sentiment-lora --train | --eval | \"<sentence>\"");
    Ok(())
}
