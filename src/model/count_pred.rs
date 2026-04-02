//! Count prediction layer for GLiNER2.
//!
//! Implements a 2-layer MLP: hidden_size → hidden_size*2 → max_count
//! Architecture: Linear(hidden_size, hidden_size*2) + ReLU + Linear(hidden_size*2, max_count)

use candle_core::{Device, DType, Tensor};
use candle_nn::{Linear, VarBuilder, Module};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the count prediction layer.
#[derive(Clone)]
pub struct CountPredictionOutput {
    pub logits: Tensor,
    pub count: usize,
}

impl std::fmt::Debug for CountPredictionOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountPredictionOutput")
            .field("count", &self.count)
            .field("logits_dims", &self.logits.dims())
            .finish()
    }
}

/// 2-layer MLP count predictor: hidden_size → hidden_size*2 → max_count
pub struct CountPredictionLayer {
    pub hidden_size: usize,
    pub max_count: usize,
    layer1: Linear,    // hidden_size → hidden_size*2
    layer2: Linear,    // hidden_size*2 → max_count
    device: Device,
}

impl std::fmt::Debug for CountPredictionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountPredictionLayer")
            .field("hidden_size", &self.hidden_size)
            .field("max_count", &self.max_count)
            .field("device", &self.device)
            .finish()
    }
}

impl Clone for CountPredictionLayer {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            max_count: self.max_count,
            layer1: self.layer1.clone(),
            layer2: self.layer2.clone(),
            device: self.device.clone(),
        }
    }
}

impl CountPredictionLayer {
    pub fn new(hidden_size: usize, max_count: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let layer1 = candle_nn::linear(hidden_size, hidden_size * 2, vb.pp("0"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create count_pred layer1: {e}")))?;
        let layer2 = candle_nn::linear(hidden_size * 2, max_count, vb.pp("2"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create count_pred layer2: {e}")))?;

        Ok(Self { hidden_size, max_count, layer1, layer2, device })
    }

    pub fn from_config(config: &ExtractorConfig, device: Device) -> Result<Self> {
        Self::new(config.hidden_size, 20, device)
    }

    pub fn from_var_builder(vb: VarBuilder, config: &ExtractorConfig, device: Device) -> Result<Self> {
        let vb = vb.pp("count_pred");
        let max_count = 20;
        let layer1 = candle_nn::linear(config.hidden_size, config.hidden_size * 2, vb.pp("0"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to load count_pred layer1: {e}")))?;
        let layer2 = candle_nn::linear(config.hidden_size * 2, max_count, vb.pp("2"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to load count_pred layer2: {e}")))?;

        Ok(Self { hidden_size: config.hidden_size, max_count, layer1, layer2, device })
    }

    pub fn predict_count(&self, schema_emb: &Tensor) -> Result<CountPredictionOutput> {
        let x = self.layer1.forward(schema_emb)
            .map_err(|e| GlinerError::model_loading(format!("CountPred layer1 forward failed: {e}")))?;
        let x = x.relu()
            .map_err(|e| GlinerError::model_loading(format!("CountPred ReLU failed: {e}")))?;
        let logits = self.layer2.forward(&x)
            .map_err(|e| GlinerError::model_loading(format!("CountPred layer2 forward failed: {e}")))?;
        
        // Get argmax for predicted count
        let argmax = logits.argmax(1)
            .map_err(|e| GlinerError::model_loading(format!("CountPred argmax failed: {e}")))?;
        // Squeeze to scalar: [1] -> []
        let argmax = argmax.squeeze(0)
            .map_err(|e| GlinerError::model_loading(format!("CountPred squeeze failed: {e}")))?;
        let count = argmax.to_scalar::<u32>()
            .map_err(|e| GlinerError::model_loading(format!("CountPred to_scalar failed: {e}")))? as usize;

        Ok(CountPredictionOutput { logits, count })
    }

    pub fn device(&self) -> &Device { &self.device }
}
