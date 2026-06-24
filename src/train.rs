use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{
    loss,
    optim::{AdamW, Optimizer, ParamsAdamW},
};

use crate::data::SentimentSample;
use crate::model::SentimentModel;

pub fn train(
    model: &SentimentModel,
    samples: &[SentimentSample],
    device: &Device,
    epochs: usize,
    lr: f64,
) -> Result<()> {
    let mut opt = AdamW::new(
        model.var_map.all_vars(),
        ParamsAdamW { lr, ..Default::default() },
    )?;

    for epoch in 0..epochs {
        let mut total_loss = 0f32;

        for sample in samples {
            let seq_len = sample.input_ids.len();
            let input_ids = Tensor::from_vec(sample.input_ids.clone(), (1, seq_len), device)?;
            let attention_mask = Tensor::from_vec(sample.attention_mask.clone(), (1, seq_len), device)?;
            let label = Tensor::from_vec(vec![sample.label], (1,), device)?;

            let logits = model.forward(&input_ids, &attention_mask)?;
            let loss = loss::cross_entropy(&logits, &label)?;

            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;
        }

        let accuracy = evaluate(model, samples, device)?;
        println!(
            "Epoch {:2}: loss = {:.4}  accuracy = {:.1}%",
            epoch + 1,
            total_loss / samples.len() as f32,
            accuracy * 100.0,
        );
    }

    Ok(())
}

pub fn evaluate(model: &SentimentModel, samples: &[SentimentSample], device: &Device) -> Result<f32> {
    let mut correct = 0usize;

    for sample in samples {
        let seq_len = sample.input_ids.len();
        let input_ids = Tensor::from_vec(sample.input_ids.clone(), (1, seq_len), device)?;
        let attention_mask = Tensor::from_vec(sample.attention_mask.clone(), (1, seq_len), device)?;

        let logits = model.forward(&input_ids, &attention_mask)?;
        let pred = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;

        if pred == sample.label {
            correct += 1;
        }
    }

    Ok(correct as f32 / samples.len() as f32)
}
