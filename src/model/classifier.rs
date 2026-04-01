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
//! - Configurable dropout
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::model::ClassifierHead;
//! use tch::{Tensor, Device, Kind};
//!
//! let hidden_size = 768;
//! let classifier = ClassifierHead::new(hidden_size, Device::Cpu)?;
//!
//! // Schema embeddings: (num_labels, hidden_size)
//! let schema_embs = Tensor::randn([3, hidden_size], (Kind::Float, Device::Cpu));
//!
//! let logits = classifier.forward(&schema_embs)?;
//! // logits shape: (num_labels,)
//! assert_eq!(logits.size(), &[3]);
//! ```

use tch::{nn, Device, Kind, Tensor};

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
            return Ok(probs.shallow_clone());
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
            Activation::Sigmoid => Ok(self.logits.sigmoid()),
            Activation::Softmax => Ok(self.logits.softmax(-1, Kind::Float)),
            Activation::None => Ok(self.logits.shallow_clone()),
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
#[derive(Debug)]
pub struct ClassifierHead {
    /// Hidden size of the input embeddings.
    pub hidden_size: usize,
    /// Weight tensor of shape `(1, hidden_size)`.
    weight: Tensor,
    /// Bias tensor of shape `(1,)`.
    bias: Tensor,
    /// Dropout probability (0.0 = disabled).
    dropout_prob: f64,
    /// Device for tensor operations.
    device: Device,
}

impl Clone for ClassifierHead {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            weight: self.weight.shallow_clone(),
            bias: self.bias.shallow_clone(),
            dropout_prob: self.dropout_prob,
            device: self.device,
        }
    }
}

impl ClassifierHead {
    /// Create a new classifier head.
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
        // Initialize weights with small random values
        let weight = Tensor::randn(
            &[1, hidden_size as i64],
            (Kind::Float, device),
        ) * 0.02;

        let bias = Tensor::zeros(&[1], (Kind::Float, device));

        Ok(Self {
            hidden_size,
            weight,
            bias,
            dropout_prob: 0.0,
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
        let mut classifier = Self::new(config.hidden_size, device)?;
        classifier.dropout_prob = config.hidden_dropout_prob as f64;
        Ok(classifier)
    }

    /// Initialize the classifier with weights from a VarStore.
    ///
    /// # Arguments
    ///
    /// * `vs` - The variable store containing the weights.
    /// * `prefix` - The prefix for weight names.
    pub fn init_from_varstore(&mut self, vs: &nn::Path, prefix: &str) -> Result<()> {
        let cls_path = vs / prefix;

        // Load weight
        let w = cls_path.var(
            "weight",
            &[1, self.hidden_size as i64],
            nn::init::DEFAULT_KAIMING_UNIFORM,
        );
        self.weight = w;

        // Load bias
        let b = cls_path.var("bias", &[1], nn::Init::Const(0.0));
        self.bias = b;

        Ok(())
    }

    /// Set the dropout probability.
    ///
    /// # Arguments
    ///
    /// * `prob` - Dropout probability (0.0 to 1.0).
    pub fn set_dropout(&mut self, prob: f64) {
        self.dropout_prob = prob;
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
        let size = schema_embs.size();

        // Validate input dimensions
        if size.is_empty() {
            return Err(GlinerError::dimension_mismatch(
                vec![2],
                vec![0],
            ));
        }

        let hidden_dim = size[size.len() - 1] as usize;
        if hidden_dim != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(
                vec![self.hidden_size],
                vec![hidden_dim],
            ));
        }

        // Apply dropout if enabled
        let input = if self.dropout_prob > 0.0 {
            schema_embs.dropout(self.dropout_prob, true)
        } else {
            schema_embs.shallow_clone()
        };

        // Compute logits: input @ weight.T + bias
        // weight shape: (1, hidden_size)
        // input shape: (..., hidden_size)
        // output shape: (..., 1)
        let logits = input.matmul(&self.weight.transpose_copy(0, 1)) + &self.bias;

        // Squeeze the last dimension
        let logits = logits.squeeze_dim(-1);

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

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Set the weight tensor.
    pub fn set_weight(&mut self, weight: Tensor) {
        self.weight = weight;
    }

    /// Set the bias tensor.
    pub fn set_bias(&mut self, bias: Tensor) {
        self.bias = bias;
    }
}

/// Builder for constructing `ClassifierHead` with custom settings.
#[derive(Debug, Clone)]
pub struct ClassifierBuilder {
    hidden_size: usize,
    device: Device,
    dropout_prob: f64,
    activation: Activation,
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            device: Device::Cpu,
            dropout_prob: 0.0,
            activation: Activation::Auto,
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

    /// Set the dropout probability.
    pub fn dropout(mut self, prob: f64) -> Self {
        self.dropout_prob = prob;
        self
    }

    /// Set the activation function.
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Build the classifier head.
    pub fn build(self) -> Result<ClassifierHead> {
        let mut classifier = ClassifierHead::new(self.hidden_size, self.device)?;
        classifier.dropout_prob = self.dropout_prob;
        Ok(classifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let classifier = ClassifierHead::new(768, Device::Cpu);
        assert!(classifier.is_ok());
        let classifier = classifier.unwrap();
        assert_eq!(classifier.hidden_size, 768);
        assert_eq!(classifier.weight.size(), &[1, 768]);
        assert_eq!(classifier.bias.size(), &[1]);
    }

    #[test]
    fn test_classifier_forward() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu));

        let output = classifier.forward(&schema_embs);
        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.logits.size(), &[3]);
    }

    #[test]
    fn test_classifier_forward_with_activation() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu));

        // Test sigmoid activation (multi-label)
        let output = classifier.forward_with_activation(&schema_embs, Activation::Sigmoid, true);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert!(output.probs.is_some());
        let probs = output.probs.unwrap();
        assert_eq!(probs.size(), &[3]);

        // Verify probabilities are in [0, 1] range
        let probs_vec: Vec<f32> = probs.try_into().unwrap();
        for p in &probs_vec {
            assert!(*p >= 0.0 && *p <= 1.0);
        }

        // Test softmax activation (single-label)
        let output = classifier.forward_with_activation(&schema_embs, Activation::Softmax, false);
        assert!(output.is_ok());
        let output = output.unwrap();
        let probs = output.probs.unwrap();
        let probs_vec: Vec<f32> = probs.try_into().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_classifier_auto_activation() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu));

        // Multi-label should use sigmoid
        let output = classifier.forward_with_activation(&schema_embs, Activation::Auto, true);
        assert!(output.is_ok());
        let output = output.unwrap();
        let probs = output.probs.unwrap();
        let probs_vec: Vec<f32> = probs.try_into().unwrap();
        // Sigmoid outputs are independent, sum won't be 1.0
        let sum: f32 = probs_vec.iter().sum();
        assert!(sum > 0.0);

        // Single-label should use softmax
        let output = classifier.forward_with_activation(&schema_embs, Activation::Auto, false);
        assert!(output.is_ok());
        let output = output.unwrap();
        let probs = output.probs.unwrap();
        let probs_vec: Vec<f32> = probs.try_into().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_classifier_batch() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let embs_list = vec![
            Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu)),
            Tensor::randn(&[5, 768], (Kind::Float, Device::Cpu)),
            Tensor::randn(&[2, 768], (Kind::Float, Device::Cpu)),
        ];

        let outputs = classifier.forward_batch(&embs_list);
        assert!(outputs.is_ok());
        let outputs = outputs.unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].logits.size(), &[3]);
        assert_eq!(outputs[1].logits.size(), &[5]);
        assert_eq!(outputs[2].logits.size(), &[2]);
    }

    #[test]
    fn test_classifier_invalid_input() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();

        // Wrong hidden size
        let bad_embs = Tensor::randn(&[3, 512], (Kind::Float, Device::Cpu));
        assert!(classifier.forward(&bad_embs).is_err());

        // Empty tensor
        let bad_embs = Tensor::randn(&[0], (Kind::Float, Device::Cpu));
        assert!(classifier.forward(&bad_embs).is_err());
    }

    #[test]
    fn test_classifier_dropout() {
        let mut classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        classifier.set_dropout(0.5);

        let schema_embs = Tensor::ones(&[3, 768], (Kind::Float, Device::Cpu));

        // With dropout, output should vary between calls
        let output1 = classifier.forward(&schema_embs).unwrap();
        let output2 = classifier.forward(&schema_embs).unwrap();

        // Due to dropout, outputs should be different (with high probability)
        // Note: This test could theoretically fail due to randomness, but very unlikely
        let diff = (&output1.logits - &output2.logits).abs().sum(None).double_value(&[]);
        assert!(diff > 0.0);
    }

    #[test]
    fn test_classifier_builder() {
        let classifier = ClassifierBuilder::new()
            .hidden_size(512)
            .device(Device::Cpu)
            .dropout(0.3)
            .activation(Activation::Sigmoid)
            .build();

        assert!(classifier.is_ok());
        let classifier = classifier.unwrap();
        assert_eq!(classifier.hidden_size, 512);
        assert!((classifier.dropout_prob - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_activation_from_str() {
        assert_eq!("sigmoid".parse::<Activation>().unwrap(), Activation::Sigmoid);
        assert_eq!("softmax".parse::<Activation>().unwrap(), Activation::Softmax);
        assert_eq!("auto".parse::<Activation>().unwrap(), Activation::Auto);
        assert_eq!("none".parse::<Activation>().unwrap(), Activation::None);
        assert!("invalid".parse::<Activation>().is_err());
    }

    #[test]
    fn test_classifier_deterministic_without_dropout() {
        let classifier = ClassifierHead::new(768, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu));

        let output1 = classifier.forward(&schema_embs).unwrap();
        let output2 = classifier.forward(&schema_embs).unwrap();

        assert_eq!(output1.logits, output2.logits);
    }

    #[test]
    fn test_classifier_getters_setters() {
        let mut classifier = ClassifierHead::new(768, Device::Cpu).unwrap();

        let new_weight = Tensor::ones(&[1, 768], (Kind::Float, Device::Cpu));
        let new_bias = Tensor::zeros(&[1], (Kind::Float, Device::Cpu));

        classifier.set_weight(new_weight.shallow_clone());
        classifier.set_bias(new_bias.shallow_clone());

        assert_eq!(classifier.weight(), &new_weight);
        assert_eq!(classifier.bias(), &new_bias);
    }
}
