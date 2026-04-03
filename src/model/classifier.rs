//! Classifier head for GLiNER2.
//!
//! Implements a 2-layer MLP: hidden_size → hidden_size*2 → 1
//! Architecture: Linear(hidden_size, hidden_size*2) + ReLU + Linear(hidden_size*2, 1)

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the classifier head.
#[derive(Clone)]
pub struct ClassifierOutput {
    pub logits: Tensor,
}

impl std::fmt::Debug for ClassifierOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassifierOutput")
            .field("logits_dims", &self.logits.dims())
            .finish()
    }
}

/// 2-layer MLP classifier: hidden_size → hidden_size*2 → 1
pub struct ClassifierHead {
    pub hidden_size: usize,
    layer1: Linear, // hidden_size → hidden_size*2
    layer2: Linear, // hidden_size*2 → 1
    device: Device,
}

impl std::fmt::Debug for ClassifierHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassifierHead")
            .field("hidden_size", &self.hidden_size)
            .field("device", &self.device)
            .finish()
    }
}

impl Clone for ClassifierHead {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            layer1: self.layer1.clone(),
            layer2: self.layer2.clone(),
            device: self.device.clone(),
        }
    }
}

impl ClassifierHead {
    pub fn new(hidden_size: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer1 = candle_nn::linear(hidden_size, hidden_size * 2, vb.pp("0")).map_err(|e| {
            GlinerError::model_loading(format!("Failed to create classifier layer1: {e}"))
        })?;
        let layer2 = candle_nn::linear(hidden_size * 2, 1, vb.pp("2")).map_err(|e| {
            GlinerError::model_loading(format!("Failed to create classifier layer2: {e}"))
        })?;

        Ok(Self {
            hidden_size,
            layer1,
            layer2,
            device,
        })
    }

    pub fn from_config(config: &ExtractorConfig, device: Device) -> Result<Self> {
        Self::new(config.hidden_size, device)
    }

    pub fn from_var_builder(
        vb: VarBuilder,
        config: &ExtractorConfig,
        device: Device,
    ) -> Result<Self> {
        let vb = vb.pp("classifier");
        // GLiNER2 uses indices 0 and 2 for the two linear layers
        let layer1 = candle_nn::linear(config.hidden_size, config.hidden_size * 2, vb.pp("0"))
            .map_err(|e| {
                GlinerError::model_loading(format!("Failed to load classifier layer1: {e}"))
            })?;
        let layer2 = candle_nn::linear(config.hidden_size * 2, 1, vb.pp("2")).map_err(|e| {
            GlinerError::model_loading(format!("Failed to load classifier layer2: {e}"))
        })?;

        Ok(Self {
            hidden_size: config.hidden_size,
            layer1,
            layer2,
            device,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer1.forward(x).map_err(|e| {
            GlinerError::model_loading(format!("Classifier layer1 forward failed: {e}"))
        })?;
        let x = x
            .relu()
            .map_err(|e| GlinerError::model_loading(format!("Classifier ReLU failed: {e}")))?;
        let logits = self.layer2.forward(&x).map_err(|e| {
            GlinerError::model_loading(format!("Classifier layer2 forward failed: {e}"))
        })?;
        // Squeeze last dim: (..., 1) -> (...)
        let logits = logits
            .squeeze(1)
            .map_err(|e| GlinerError::model_loading(format!("Classifier squeeze failed: {e}")))?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
