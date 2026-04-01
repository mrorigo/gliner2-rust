//! Classifier head for GLiNER2.
//!
//! This module implements the classification head used for text classification
//! tasks within the GLiNER2 model. It takes schema embeddings as input and
//! produces logits for each classification label.
//!
//! # Architecture
//!
//! The classifier consists of:
//! - A linear layer: `hidden_size -> 1` (binary classification per label)
//! - Optional activation: sigmoid or softmax
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::model::ClassifierHead;
//! use candle_core::{Tensor, Device};
//!
//! let hidden_size = 768;
//! let classifier = ClassifierHead::new(hidden_size, Device::Cpu)?;
//!
//! // Schema embeddings: (num_labels, hidden_size)
//! let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, hidden_size), &Device::Cpu)?;
//!
//! let output = classifier.forward(&schema_embs)?;
//! // logits shape: (3,)
//! assert_eq!(output.logits.dims()?, &[3]);
//! ```

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Activation function for the classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Sigmoid activation (for multi-label classification).
    Sigmoid,
    /// Softmax activation (for single-label classification).
    Softmax,
    /// Automatic: sigmoid for multi-label, softmax for single-label.
    Auto,
    /// No activation (raw logits).
    None,
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::Softmax => write!(f, "softmax"),
            Activation::Auto => write!(f, "auto"),
            Activation::None => write!(f, "none"),
        }
    }
}

impl std::str::FromStr for Activation {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "sigmoid" => Ok(Activation::Sigmoid),
            "softmax" => Ok(Activation::Softmax),
            "auto" => Ok(Activation::Auto),
            "none" => Ok(Activation::None),
            _ => Err(GlinerError::config(format!(
                "Unknown activation: {s}. Expected 'sigmoid', 'softmax', 'auto', or 'none'"
            ))),
        }
    }
}

/// Output of the classifier head.
#[derive(Debug)]
pub struct ClassifierOutput {
    /// Raw logits of shape `(num_labels,)` or `(batch_size, num_labels)`.
    pub logits: Tensor,
    /// Probabilities after activation (if applied).
    pub probs: Option<Tensor>,
}

impl ClassifierOutput {
    /// Create a new classifier output.
    pub fn new(logits: Tensor, probs: Option<Tensor>) -> Self {
        Self { logits, probs }
    }

    /// Get the probabilities, computing them from logits if not already computed.
    ///
    /// # Arguments
    ///
    /// * `activation` - The activation function to use.
    /// * `is_multi_label` - Whether this is multi-label classification.
    pub fn get_probs(&self, activation: Activation, is_multi_label: bool) -> Result<Tensor> {
        if let Some(ref probs) = self.probs {
            return Ok(probs.clone());
        }

        let effective_activation = match activation {
            Activation::Auto => {
                if is_multi_label {
                    Activation::Sigmoid
                } else {
                    Activation::Softmax
                }
            }
            other => other,
        };

        match effective_activation {
            Activation::Sigmoid => candle_nn::ops::sigmoid(&self.logits).map_err(|e| GlinerError::model_loading(format!("Sigmoid activation failed: {e}"))),
            Activation::Softmax => {
                // Logits are squeezed to 1D (num_labels,), so softmax on dim 0
                let dim = if self.logits.dims().len() == 1 { 0 } else { 1 };
                candle_nn::ops::softmax(&self.logits, dim).map_err(|e| GlinerError::model_loading(format!("Softmax activation failed: {e}")))
            }
            Activation::None => Ok(self.logits.clone()),
            Activation::Auto => unreachable!(),
        }
    }
}

/// Classifier head for GLiNER2 classification tasks.
///
/// This module implements a linear classification head that takes schema
/// embeddings and produces logits for each label. It supports both
/// single-label and multi-label classification with configurable activation.
///
/// # Tensor Shapes
///
/// - Input: `(num_labels, hidden_size)` or `(batch_size, num_labels, hidden_size)`
/// - Output logits: `(num_labels,)` or `(batch_size, num_labels)`
pub struct ClassifierHead {
    /// Hidden size of the input embeddings.
    pub hidden_size: usize,
    /// Linear layer (weight: (1, hidden_size), bias: (1,)).
    linear: Linear,
    /// Device for tensor operations.
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

impl ClassifierHead {
    /// Create a new classifier head with random initialization.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The hidden size of the input embeddings.
    /// * `device` - The device to place tensors on.
    ///
    /// # Returns
    ///
    /// A new `ClassifierHead` with randomly initialized weights.
    pub fn new(hidden_size: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Create linear layer: hidden_size -> 1
        let linear = candle_nn::linear(hidden_size, 1, vb.pp("classifier"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create classifier linear layer: {e}")))?;

        Ok(Self {
            hidden_size,
            linear,
            device,
        })
    }

    /// Create a classifier head from a config.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `device` - The device to place tensors on.
    pub fn from_config(config: &ExtractorConfig, device: Device) -> Result<Self> {
        Self::new(config.hidden_size, device)
    }

    /// Create a classifier head loaded from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `vb` - The VarBuilder containing the weights.
    /// * `config` - The extractor configuration.
    /// * `device` - The device to place tensors on.
    pub fn from_var_builder(
        vb: VarBuilder,
        config: &ExtractorConfig,
        device: Device,
    ) -> Result<Self> {
        let linear = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to load classifier weights: {e}")))?;

        Ok(Self {
            hidden_size: config.hidden_size,
            linear,
            device,
        })
    }

    /// Forward pass: compute classification logits.
    ///
    /// # Arguments
    ///
    /// * `schema_embs` - Schema embeddings of shape `(num_labels, hidden_size)`.
    ///
    /// # Returns
    ///
    /// A `ClassifierOutput` containing logits and optionally probabilities.
    pub fn forward(&self, schema_embs: &Tensor) -> Result<ClassifierOutput> {
        let dims = schema_embs.dims();

        if dims.is_empty() {
            return Err(GlinerError::dimension_mismatch(vec![2], vec![0]));
        }

        let hidden_dim = dims[dims.len() - 1];
        if hidden_dim != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(
                vec![self.hidden_size],
                vec![hidden_dim],
            ));
        }

        // Compute logits: linear forward -> (..., 1)
        let logits = self.linear.forward(schema_embs).map_err(|e| {
            GlinerError::model_loading(format!("Classifier forward pass failed: {e}"))
        })?;

        // Squeeze the last dimension: (..., 1) -> (...)
        let logits = logits.squeeze(1).map_err(|e| {
            GlinerError::model_loading(format!("Failed to squeeze classifier output: {e}"))
        })?;

        Ok(ClassifierOutput::new(logits, None))
    }

    /// Forward pass with activation applied.
    ///
    /// # Arguments
    ///
    /// * `schema_embs` - Schema embeddings of shape `(num_labels, hidden_size)`.
    /// * `activation` - The activation function to apply.
    /// * `is_multi_label` - Whether this is multi-label classification.
    ///
    /// # Returns
    ///
    /// A `ClassifierOutput` containing both logits and probabilities.
    pub fn forward_with_activation(
        &self,
        schema_embs: &Tensor,
        activation: Activation,
        is_multi_label: bool,
    ) -> Result<ClassifierOutput> {
        let output = self.forward(schema_embs)?;
        let probs = output.get_probs(activation, is_multi_label)?;
        Ok(ClassifierOutput::new(output.logits, Some(probs)))
    }

    /// Forward pass for a batch of schema embeddings.
    ///
    /// # Arguments
    ///
    /// * `schema_embs_list` - List of schema embeddings, each of shape `(num_labels, hidden_size)`.
    ///
    /// # Returns
    ///
    /// A vector of `ClassifierOutput`, one for each input.
    pub fn forward_batch(&self, schema_embs_list: &[Tensor]) -> Result<Vec<ClassifierOutput>> {
        schema_embs_list
            .iter()
            .map(|embs| self.forward(embs))
            .collect()
    }

    /// Get the device the classifier is on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Builder for constructing `ClassifierHead` with custom settings.
#[derive(Debug, Clone)]
pub struct ClassifierBuilder {
    hidden_size: usize,
    device: Device,
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            device: Device::Cpu,
        }
    }
}

impl ClassifierBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hidden size.
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Build the classifier head.
    pub fn build(self) -> Result<ClassifierHead> {
        ClassifierHead::new(self.hidden_size, self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_classifier_creation() {
        let classifier = ClassifierHead::new(768, Device::Cpu);
        assert!(classifier.is_ok());
        let classifier = classifier.unwrap();
        assert_eq!(classifier.hidden_size, 768);
    }

    #[test]
    fn test_classifier_forward() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap();

        let output = classifier.forward(&schema_embs);
        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.logits.dims(), &[3]);
    }

    #[test]
    fn test_classifier_forward_with_activation() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap();

        // Test sigmoid activation (multi-label)
        let output = classifier.forward_with_activation(&schema_embs, Activation::Sigmoid, true);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert!(output.probs.is_some());
        let probs = output.probs.unwrap();
        assert_eq!(probs.dims(), &[3]);

        // Verify probabilities are in [0, 1] range
        let probs_vec: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        for p in &probs_vec {
            assert!(*p >= 0.0 && *p <= 1.0);
        }

        // Test softmax activation (single-label)
        let output = classifier.forward_with_activation(&schema_embs, Activation::Softmax, false);
        assert!(output.is_ok());
        let output = output.unwrap();
        let probs = output.probs.unwrap();
        let probs_vec: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_classifier_auto_activation() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap();

        // Multi-label should use sigmoid
        let output = classifier.forward_with_activation(&schema_embs, Activation::Auto, true);
        assert!(output.is_ok());
        let output = output.unwrap();
        let probs = output.probs.unwrap();
        let probs_vec: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!(sum > 0.0);

        // Single-label should use softmax
        let output = classifier.forward_with_activation(&schema_embs, Activation::Auto, false);
        assert!(output.is_ok());
        let output = output.unwrap();
        let probs = output.probs.unwrap();
        let probs_vec: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_classifier_batch() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let embs_list = vec![
            Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap(),
            Tensor::randn(0.0f32, 1.0f32, (5, 768), &Device::Cpu).unwrap(),
            Tensor::randn(0.0f32, 1.0f32, (2, 768), &Device::Cpu).unwrap(),
        ];

        let outputs = classifier.forward_batch(&embs_list);
        assert!(outputs.is_ok());
        let outputs = outputs.unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].logits.dims(), &[3]);
        assert_eq!(outputs[1].logits.dims(), &[5]);
        assert_eq!(outputs[2].logits.dims(), &[2]);
    }

    #[test]
    fn test_classifier_invalid_input() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();

        // Wrong hidden size
        let bad_embs = Tensor::randn(0.0f32, 1.0f32, (3, 512), &Device::Cpu).unwrap();
        assert!(classifier.forward(&bad_embs).is_err());

        // Empty tensor
        let bad_embs = Tensor::zeros((0,), DType::F32, &Device::Cpu).unwrap();
        assert!(classifier.forward(&bad_embs).is_err());
    }
}
