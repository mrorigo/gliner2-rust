//! Configuration for the GLiNER2 Extractor model.
//!
//! This module defines the `ExtractorConfig` struct and related types
//! for configuring model architecture, inference behavior, and performance settings.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{GlinerError, Result};

/// Default model name for GLiNER2 base model.
pub const DEFAULT_MODEL_NAME: &str = "fastino/gliner2-base-v1";

/// Default maximum span width for entity/relation extraction.
pub const DEFAULT_MAX_WIDTH: usize = 8;

/// Default counting layer type.
pub const DEFAULT_COUNTING_LAYER: &str = "count_lstm";

/// Default token pooling strategy.
pub const DEFAULT_TOKEN_POOLING: &str = "first";

/// Default hidden size for base model.
pub const DEFAULT_HIDDEN_SIZE: usize = 768;

/// Default number of hidden layers.
pub const DEFAULT_NUM_HIDDEN_LAYERS: usize = 12;

/// Default number of attention heads.
pub const DEFAULT_NUM_ATTENTION_HEADS: usize = 12;

/// Default intermediate size (FFN hidden size).
pub const DEFAULT_INTERMEDIATE_SIZE: usize = 3072;

/// Default vocabulary size (BERT-base).
pub const DEFAULT_VOCAB_SIZE: usize = 30522;

/// Default maximum position embeddings.
pub const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 512;

/// Default layer norm epsilon.
pub const DEFAULT_LAYER_NORM_EPS: f32 = 1e-12;

/// Default hidden dropout probability.
pub const DEFAULT_HIDDEN_DROPOUT_PROB: f32 = 0.1;

/// Default attention dropout probability.
pub const DEFAULT_ATTENTION_DROPOUT_PROB: f32 = 0.1;

/// Counting layer types supported by GLiNER2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountingLayerType {
    /// LSTM-based counting layer.
    CountLstm,
    /// Linear-based counting layer.
    Linear,
}

impl std::fmt::Display for CountingLayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CountingLayerType::CountLstm => write!(f, "count_lstm"),
            CountingLayerType::Linear => write!(f, "linear"),
        }
    }
}

impl std::str::FromStr for CountingLayerType {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "count_lstm" => Ok(CountingLayerType::CountLstm),
            "linear" => Ok(CountingLayerType::Linear),
            _ => Err(GlinerError::config(format!(
                "Unknown counting layer type: {s}. Expected 'count_lstm' or 'linear'"
            ))),
        }
    }
}

/// Token pooling strategies for extracting token embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenPoolingStrategy {
    /// Use the first token's embedding.
    First,
    /// Use the last token's embedding.
    Last,
    /// Average all token embeddings.
    Mean,
    /// Use max pooling over all token embeddings.
    Max,
}

impl std::fmt::Display for TokenPoolingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenPoolingStrategy::First => write!(f, "first"),
            TokenPoolingStrategy::Last => write!(f, "last"),
            TokenPoolingStrategy::Mean => write!(f, "mean"),
            TokenPoolingStrategy::Max => write!(f, "max"),
        }
    }
}

impl std::str::FromStr for TokenPoolingStrategy {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "first" => Ok(TokenPoolingStrategy::First),
            "last" => Ok(TokenPoolingStrategy::Last),
            "mean" => Ok(TokenPoolingStrategy::Mean),
            "max" => Ok(TokenPoolingStrategy::Max),
            _ => Err(GlinerError::config(format!(
                "Unknown token pooling strategy: {s}. Expected 'first', 'last', 'mean', or 'max'"
            ))),
        }
    }
}

/// Hidden activation functions supported by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenActivation {
    /// GELU activation (original version).
    Gelu,
    /// GELU activation (approximate version).
    GeluApproximate,
    /// ReLU activation.
    Relu,
    /// SiLU/Swish activation.
    Silu,
}

impl std::fmt::Display for HiddenActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HiddenActivation::Gelu => write!(f, "gelu"),
            HiddenActivation::GeluApproximate => write!(f, "gelu_approximate"),
            HiddenActivation::Relu => write!(f, "relu"),
            HiddenActivation::Silu => write!(f, "silu"),
        }
    }
}

impl std::str::FromStr for HiddenActivation {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "gelu" => Ok(HiddenActivation::Gelu),
            "gelu_approximate" => Ok(HiddenActivation::GeluApproximate),
            "relu" => Ok(HiddenActivation::Relu),
            "silu" => Ok(HiddenActivation::Silu),
            _ => Err(GlinerError::config(format!(
                "Unknown activation function: {s}"
            ))),
        }
    }
}

/// Configuration for the GLiNER2 Extractor model.
///
/// This struct contains all parameters needed to initialize and configure
/// the GLiNER2 model for inference, including architecture parameters,
/// tokenizer settings, and performance options.
///
/// # Example
///
/// ```
/// use gliner2_rust::config::ExtractorConfig;
///
/// // Create config with defaults for base model
/// let config = ExtractorConfig::default();
///
/// // Create config for a specific model
/// let config = ExtractorConfig::new("bert-base-uncased");
///
/// // Create config with custom settings
/// let config = ExtractorConfig::builder()
///     .model_name("bert-base-uncased")
///     .max_width(12)
///     .max_len(384)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractorConfig {
    // -------------------------------------------------------------------------
    // Model Architecture
    // -------------------------------------------------------------------------
    /// Base model name or path (e.g., "bert-base-uncased").
    pub model_name: String,

    /// Hidden size of the transformer model.
    pub hidden_size: usize,

    /// Number of hidden layers in the transformer.
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Intermediate size of the feed-forward network.
    pub intermediate_size: usize,

    /// Hidden activation function.
    pub hidden_act: HiddenActivation,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Maximum number of position embeddings.
    pub max_position_embeddings: usize,

    /// Type vocabulary size (for token type IDs).
    pub type_vocab_size: usize,

    /// Padding token ID.
    pub pad_token_id: usize,

    /// Layer normalization epsilon.
    pub layer_norm_eps: f32,

    /// Hidden dropout probability.
    pub hidden_dropout_prob: f32,

    /// Attention dropout probability.
    pub attention_probs_dropout_prob: f32,

    // -------------------------------------------------------------------------
    // GLiNER2-Specific Settings
    // -------------------------------------------------------------------------
    /// Maximum span width for entity/relation extraction.
    pub max_width: usize,

    /// Type of counting layer to use.
    pub counting_layer: CountingLayerType,

    /// Token pooling strategy for schema embeddings.
    pub token_pooling: TokenPoolingStrategy,

    /// Maximum number of tokens to process (None = no limit).
    pub max_len: Option<usize>,

    // -------------------------------------------------------------------------
    // Inference Settings
    // -------------------------------------------------------------------------
    /// Whether to use FP16 precision for inference.
    pub use_fp16: bool,

    /// Whether to use BF16 precision for inference.
    pub use_bf16: bool,

    /// Device to run inference on ("cpu", "cuda", etc.).
    pub device: String,

    /// Whether to compile the model for faster inference.
    pub compile: bool,

    // -------------------------------------------------------------------------
    // Paths (for local model loading)
    // -------------------------------------------------------------------------
    /// Path to the model weights (safetensors or pytorch_model.bin).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,

    /// Path to the tokenizer files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<PathBuf>,

    /// Path to the config file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<PathBuf>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            // Model Architecture (BERT-base defaults)
            model_name: DEFAULT_MODEL_NAME.to_string(),
            hidden_size: DEFAULT_HIDDEN_SIZE,
            num_hidden_layers: DEFAULT_NUM_HIDDEN_LAYERS,
            num_attention_heads: DEFAULT_NUM_ATTENTION_HEADS,
            intermediate_size: DEFAULT_INTERMEDIATE_SIZE,
            hidden_act: HiddenActivation::Gelu,
            vocab_size: DEFAULT_VOCAB_SIZE,
            max_position_embeddings: DEFAULT_MAX_POSITION_EMBEDDINGS,
            type_vocab_size: 2,
            pad_token_id: 0,
            layer_norm_eps: DEFAULT_LAYER_NORM_EPS,
            hidden_dropout_prob: DEFAULT_HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob: DEFAULT_ATTENTION_DROPOUT_PROB,

            // GLiNER2-Specific Settings
            max_width: DEFAULT_MAX_WIDTH,
            counting_layer: CountingLayerType::CountLstm,
            token_pooling: TokenPoolingStrategy::First,
            max_len: None,

            // Inference Settings
            use_fp16: false,
            use_bf16: false,
            device: "cpu".to_string(),
            compile: false,

            // Paths
            model_path: None,
            tokenizer_path: None,
            config_path: None,
        }
    }
}

impl ExtractorConfig {
    /// Create a new config with the specified model name and default settings.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Create a config builder for fluent configuration.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// Load config from a JSON file.
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)
            .map_err(|e| GlinerError::io(&path, e))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| GlinerError::config(format!("Failed to parse config file: {e}")))?;
        Ok(config)
    }

    /// Save config to a JSON file.
    pub fn to_file(&self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| GlinerError::serialization(format!("Failed to serialize config: {e}")))?;
        std::fs::write(&path, content)
            .map_err(|e| GlinerError::io(&path, e))?;
        Ok(())
    }

    /// Get the effective dtype for inference.
    pub fn dtype(&self) -> &'static str {
        if self.use_bf16 {
            "bf16"
        } else if self.use_fp16 {
            "fp16"
        } else {
            "fp32"
        }
    }

    /// Check if quantization is enabled.
    pub fn is_quantized(&self) -> bool {
        self.use_fp16 || self.use_bf16
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(GlinerError::config("hidden_size must be > 0"));
        }
        if self.num_hidden_layers == 0 {
            return Err(GlinerError::config("num_hidden_layers must be > 0"));
        }
        if self.num_attention_heads == 0 {
            return Err(GlinerError::config("num_attention_heads must be > 0"));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(GlinerError::config(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        if self.max_width == 0 {
            return Err(GlinerError::config("max_width must be > 0"));
        }
        if self.vocab_size == 0 {
            return Err(GlinerError::config("vocab_size must be > 0"));
        }
        if self.max_position_embeddings == 0 {
            return Err(GlinerError::config("max_position_embeddings must be > 0"));
        }
        if self.use_fp16 && self.use_bf16 {
            return Err(GlinerError::config(
                "Cannot use both fp16 and bf16 simultaneously",
            ));
        }
        if let Some(max_len) = self.max_len {
            if max_len == 0 {
                return Err(GlinerError::config("max_len must be > 0 or None"));
            }
            if max_len > self.max_position_embeddings {
                return Err(GlinerError::config(format!(
                    "max_len ({max_len}) exceeds max_position_embeddings ({})",
                    self.max_position_embeddings
                )));
            }
        }
        Ok(())
    }
}

/// Builder for constructing `ExtractorConfig` with a fluent API.
#[derive(Debug, Clone, Default)]
pub struct ConfigBuilder {
    config: ExtractorConfig,
}

impl ConfigBuilder {
    /// Set the model name.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the hidden size.
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.config.hidden_size = size;
        self
    }

    /// Set the number of hidden layers.
    pub fn num_hidden_layers(mut self, layers: usize) -> Self {
        self.config.num_hidden_layers = layers;
        self
    }

    /// Set the number of attention heads.
    pub fn num_attention_heads(mut self, heads: usize) -> Self {
        self.config.num_attention_heads = heads;
        self
    }

    /// Set the intermediate size.
    pub fn intermediate_size(mut self, size: usize) -> Self {
        self.config.intermediate_size = size;
        self
    }

    /// Set the hidden activation function.
    pub fn hidden_act(mut self, act: HiddenActivation) -> Self {
        self.config.hidden_act = act;
        self
    }

    /// Set the vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set the maximum position embeddings.
    pub fn max_position_embeddings(mut self, size: usize) -> Self {
        self.config.max_position_embeddings = size;
        self
    }

    /// Set the maximum span width.
    pub fn max_width(mut self, width: usize) -> Self {
        self.config.max_width = width;
        self
    }

    /// Set the counting layer type.
    pub fn counting_layer(mut self, layer: CountingLayerType) -> Self {
        self.config.counting_layer = layer;
        self
    }

    /// Set the token pooling strategy.
    pub fn token_pooling(mut self, strategy: TokenPoolingStrategy) -> Self {
        self.config.token_pooling = strategy;
        self
    }

    /// Set the maximum token length.
    pub fn max_len(mut self, max_len: Option<usize>) -> Self {
        self.config.max_len = max_len;
        self
    }

    /// Enable FP16 precision.
    pub fn fp16(mut self, enabled: bool) -> Self {
        self.config.use_fp16 = enabled;
        if enabled {
            self.config.use_bf16 = false;
        }
        self
    }

    /// Enable BF16 precision.
    pub fn bf16(mut self, enabled: bool) -> Self {
        self.config.use_bf16 = enabled;
        if enabled {
            self.config.use_fp16 = false;
        }
        self
    }

    /// Set the device.
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }

    /// Enable model compilation.
    pub fn compile(mut self, enabled: bool) -> Self {
        self.config.compile = enabled;
        self
    }

    /// Set the model path.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Set the tokenizer path.
    pub fn tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    /// Build the config, validating all settings.
    pub fn build(self) -> Result<ExtractorConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Predefined configurations for common models.
pub mod presets {
    use super::*;

    /// GLiNER2 base model configuration (205M parameters).
    pub fn gliner2_base() -> ExtractorConfig {
        ExtractorConfig {
            model_name: "fastino/gliner2-base-v1".to_string(),
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenActivation::Gelu,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            pad_token_id: 0,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_width: 8,
            counting_layer: CountingLayerType::CountLstm,
            token_pooling: TokenPoolingStrategy::First,
            max_len: None,
            use_fp16: false,
            use_bf16: false,
            device: "cpu".to_string(),
            compile: false,
            model_path: None,
            tokenizer_path: None,
            config_path: None,
        }
    }

    /// GLiNER2 large model configuration (340M parameters).
    pub fn gliner2_large() -> ExtractorConfig {
        ExtractorConfig {
            model_name: "fastino/gliner2-large-v1".to_string(),
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: HiddenActivation::Gelu,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            pad_token_id: 0,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_width: 8,
            counting_layer: CountingLayerType::CountLstm,
            token_pooling: TokenPoolingStrategy::First,
            max_len: None,
            use_fp16: false,
            use_bf16: false,
            device: "cpu".to_string(),
            compile: false,
            model_path: None,
            tokenizer_path: None,
            config_path: None,
        }
    }

    /// GLiNER2 base model with FP16 for faster inference.
    pub fn gliner2_base_fp16() -> ExtractorConfig {
        let mut config = gliner2_base();
        config.use_fp16 = true;
        config
    }

    /// GLiNER2 base model optimized for CPU inference.
    pub fn gliner2_base_cpu_optimized() -> ExtractorConfig {
        let mut config = gliner2_base();
        config.max_len = Some(384);
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExtractorConfig::default();
        assert_eq!(config.model_name, DEFAULT_MODEL_NAME);
        assert_eq!(config.hidden_size, DEFAULT_HIDDEN_SIZE);
        assert_eq!(config.max_width, DEFAULT_MAX_WIDTH);
        assert_eq!(config.counting_layer, CountingLayerType::CountLstm);
        assert_eq!(config.token_pooling, TokenPoolingStrategy::First);
        assert!(config.max_len.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = ExtractorConfig::builder()
            .model_name("bert-base-uncased")
            .max_width(12)
            .max_len(Some(384))
            .fp16(true)
            .build()
            .unwrap();

        assert_eq!(config.model_name, "bert-base-uncased");
        assert_eq!(config.max_width, 12);
        assert_eq!(config.max_len, Some(384));
        assert!(config.use_fp16);
        assert!(!config.use_bf16);
    }

    #[test]
    fn test_config_validation() {
        let config = ExtractorConfig::builder()
            .hidden_size(0)
            .build();
        assert!(config.is_err());

        let config = ExtractorConfig::builder()
            .fp16(true)
            .bf16(true)
            .build();
        assert!(config.is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = ExtractorConfig::builder()
            .model_name("test-model")
            .max_len(Some(256))
            .build()
            .unwrap();

        let json = serde_json::to_string(&config).unwrap();
        let loaded: ExtractorConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.model_name, "test-model");
        assert_eq!(loaded.max_len, Some(256));
    }

    #[test]
    fn test_presets() {
        let base = presets::gliner2_base();
        assert_eq!(base.hidden_size, 768);
        assert_eq!(base.num_hidden_layers, 12);

        let large = presets::gliner2_large();
        assert_eq!(large.hidden_size, 1024);
        assert_eq!(large.num_hidden_layers, 24);

        let fp16 = presets::gliner2_base_fp16();
        assert!(fp16.use_fp16);

        let cpu_opt = presets::gliner2_base_cpu_optimized();
        assert_eq!(cpu_opt.max_len, Some(384));
    }

    #[test]
    fn test_counting_layer_from_str() {
        assert_eq!(
            "count_lstm".parse::<CountingLayerType>().unwrap(),
            CountingLayerType::CountLstm
        );
        assert_eq!(
            "linear".parse::<CountingLayerType>().unwrap(),
            CountingLayerType::Linear
        );
        assert!("invalid".parse::<CountingLayerType>().is_err());
    }

    #[test]
    fn test_token_pooling_from_str() {
        assert_eq!(
            "first".parse::<TokenPoolingStrategy>().unwrap(),
            TokenPoolingStrategy::First
        );
        assert_eq!(
            "mean".parse::<TokenPoolingStrategy>().unwrap(),
            TokenPoolingStrategy::Mean
        );
        assert!("invalid".parse::<TokenPoolingStrategy>().is_err());
    }
}
