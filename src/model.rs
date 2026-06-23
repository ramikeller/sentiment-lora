use anyhow::{Context, Result};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use std::path::PathBuf;

// Everything the rest of the program needs after model loading.
pub struct LoadedBert {
    pub model: BertModel,
    pub tokenizer_path: PathBuf,
    pub device: Device,
}

// Download (or load from cache) the BERT weights and config from Hugging Face,
// then construct the BertModel ready for forward passes.
//
// model_id: a HF repo name, e.g. "bert-base-uncased"
// device:   Device::Cpu for now — we add CUDA support later if needed
pub fn load_bert(model_id: &str, device: &Device) -> Result<LoadedBert> {
    // hf-hub's sync API handles caching automatically.
    // Files land in ~/.cache/huggingface/hub/ on first run,
    // then are reused on subsequent runs without a network call.
    let api = Api::new().context("failed to create HF Hub client")?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    println!("Fetching model files for '{model_id}' ...");

    // config.json describes the architecture (layer count, hidden size, etc.)
    let config_path = repo.get("config.json").context("failed to fetch config.json")?;

    // tokenizer.json is what data.rs uses to convert text → token IDs
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("failed to fetch tokenizer.json")?;

    // model.safetensors holds the pre-trained weights (~440 MB for bert-base)
    let weights_path = repo
        .get("model.safetensors")
        .context("failed to fetch model.safetensors")?;

    // Deserialize config.json into BertConfig using serde_json.
    // BertConfig tells candle-transformers exactly how to build the model graph.
    let config_str = std::fs::read_to_string(&config_path)
        .context("could not read config.json")?;
    let bert_config: BertConfig =
        serde_json::from_str(&config_str).context("could not parse config.json")?;

    // Load tensors and remap old TF-style LayerNorm key names (gamma/beta) to
    // the PyTorch-style names (weight/bias) that candle-transformers expects.
    // Older bert-base-uncased snapshots use the TF convention.
    let raw_tensors = candle_core::safetensors::load(&weights_path, device)
        .context("failed to load model weights")?;
    let tensors: HashMap<String, _> = raw_tensors
        .into_iter()
        .map(|(k, v)| (k.replace("gamma", "weight").replace("beta", "bias"), v))
        .collect();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

    // Build the BERT model graph and populate it with the loaded weights.
    // .pp("bert") scopes all tensor lookups under the "bert." prefix,
    // matching how HuggingFace names weights: "bert.embeddings.word_embeddings.weight" etc.
    let model = BertModel::load(vb.pp("bert"), &bert_config).context("failed to build BERT model")?;

    println!("BERT model loaded successfully.");

    Ok(LoadedBert {
        model,
        tokenizer_path,
        device: device.clone(),
    })
}
