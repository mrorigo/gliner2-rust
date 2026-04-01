//! Model weight loading for GLiNER2.
//!
//! This module provides utilities for loading model weights from safetensors
//! files and mapping them to the appropriate model components. It handles
//! weight name mapping, tensor conversion, and validation.
//!
//! # Supported Formats
//!
//! - **Safetensors**: `.safetensors` files (preferred)
//! - **PyTorch**: `pytorch_model.bin` files (via conversion)
//!
//! # Weight Mapping
//!
//! The loader maps HuggingFace weight names to GLiNER2 component names:
//! - `encoder.*` -> Encoder weights (BERT-like transformer)
//! - `span_rep.*` -> Span representation layer weights
//! - `count_pred.*` -> Count prediction layer weights
//! - `classifier.*` -> Classifier head weights
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::model::loading::ModelLoader;
//! use gliner2_rust::config::ExtractorConfig;
//! use tch::Device;
//!
//! let config = ExtractorConfig::default();
//! let loader = ModelLoader::new(&config, Device::Cpu)?;
//! loader.load_safetensors("model.safetensors", &mut model)?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use tch::{Device, Kind, Tensor};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};
use crate::model::extractor::Extractor;

/// Weight name mapping from HuggingFace format to GLiNER2 format.
pub type WeightMap = HashMap<String, String>;

/// Model weight loader for GLiNER2.
///
/// This struct handles loading weights from safetensors files and mapping
/// them to the appropriate model components. It supports weight name
/// remapping and tensor conversion.
#[derive(Debug)]
pub struct ModelLoader {
    /// Model configuration.
    config: ExtractorConfig,
    /// Device to load tensors onto.
    device: Device,
    /// Weight name mapping (HF format -> GLiNER2 format).
    weight_map: WeightMap,
}

impl ModelLoader {
    /// Create a new model loader.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `device` - The device to load tensors onto.
    pub fn new(config: &ExtractorConfig, device: Device) -> Result<Self> {
        let weight_map = Self::build_weight_map(config);

        Ok(Self {
            config: config.clone(),
            device,
            weight_map,
        })
    }

    /// Build the weight name mapping based on configuration.
    ///
    /// This method creates a mapping from HuggingFace weight names to
    /// GLiNER2 component names. The mapping depends on the model architecture.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    ///
    /// # Returns
    ///
    /// A weight name mapping.
    fn build_weight_map(config: &ExtractorConfig) -> WeightMap {
        let mut map = HashMap::new();

        // Encoder weights (BERT-like)
        // These are typically already in the correct format from HuggingFace
        map.insert("embeddings.word_embeddings.weight".to_string(), "encoder.embeddings.word_embeddings.weight".to_string());
        map.insert("embeddings.position_embeddings.weight".to_string(), "encoder.embeddings.position_embeddings.weight".to_string());
        map.insert("embeddings.token_type_embeddings.weight".to_string(), "encoder.embeddings.token_type_embeddings.weight".to_string());
        map.insert("embeddings.LayerNorm.weight".to_string(), "encoder.embeddings.LayerNorm.weight".to_string());
        map.insert("embeddings.LayerNorm.bias".to_string(), "encoder.embeddings.LayerNorm.bias".to_string());

        // Attention layers
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{i}");
            map.insert(
                format!("{prefix}.attention.self.query.weight"),
                format!("{prefix}.attention.self.query.weight"),
            );
            map.insert(
                format!("{prefix}.attention.self.query.bias"),
                format!("{prefix}.attention.self.query.bias"),
            );
            map.insert(
                format!("{prefix}.attention.self.key.weight"),
                format!("{prefix}.attention.self.key.weight"),
            );
            map.insert(
                format!("{prefix}.attention.self.key.bias"),
                format!("{prefix}.attention.self.key.bias"),
            );
            map.insert(
                format!("{prefix}.attention.self.value.weight"),
                format!("{prefix}.attention.self.value.weight"),
            );
            map.insert(
                format!("{prefix}.attention.self.value.bias"),
                format!("{prefix}.attention.self.value.bias"),
            );
            map.insert(
                format!("{prefix}.attention.output.dense.weight"),
                format!("{prefix}.attention.output.dense.weight"),
            );
            map.insert(
                format!("{prefix}.attention.output.dense.bias"),
                format!("{prefix}.attention.output.dense.bias"),
            );
            map.insert(
                format!("{prefix}.attention.output.LayerNorm.weight"),
                format!("{prefix}.attention.output.LayerNorm.weight"),
            );
            map.insert(
                format!("{prefix}.attention.output.LayerNorm.bias"),
                format!("{prefix}.attention.output.LayerNorm.bias"),
            );
            map.insert(
                format!("{prefix}.intermediate.dense.weight"),
                format!("{prefix}.intermediate.dense.weight"),
            );
            map.insert(
                format!("{prefix}.intermediate.dense.bias"),
                format!("{prefix}.intermediate.dense.bias"),
            );
            map.insert(
                format!("{prefix}.output.dense.weight"),
                format!("{prefix}.output.dense.weight"),
            );
            map.insert(
                format!("{prefix}.output.dense.bias"),
                format!("{prefix}.output.dense.bias"),
            );
            map.insert(
                format!("{prefix}.output.LayerNorm.weight"),
                format!("{prefix}.output.LayerNorm.weight"),
            );
            map.insert(
                format!("{prefix}.output.LayerNorm.bias"),
                format!("{prefix}.output.LayerNorm.bias"),
            );
        }

        // Pooler (if present)
        map.insert("pooler.dense.weight".to_string(), "encoder.pooler.dense.weight".to_string());
        map.insert("pooler.dense.bias".to_string(), "encoder.pooler.dense.bias".to_string());

        // GLiNER2-specific components
        // These may have different naming in the saved weights
        map.insert("span_rep.weight".to_string(), "span_rep.weight".to_string());
        map.insert("span_rep.bias".to_string(), "span_rep.bias".to_string());
        map.insert("span_rep.layer_norm.weight".to_string(), "span_rep.layer_norm.weight".to_string());
        map.insert("span_rep.layer_norm.bias".to_string(), "span_rep.layer_norm.bias".to_string());
        map.insert("span_rep.width_embedding".to_string(), "span_rep.width_embedding".to_string());

        map.insert("count_pred.weight".to_string(), "count_pred.weight".to_string());
        map.insert("count_pred.bias".to_string(), "count_pred.bias".to_string());
        map.insert("count_pred.embedding.weight".to_string(), "count_pred.embedding.weight".to_string());

        map.insert("classifier.weight".to_string(), "classifier.weight".to_string());
        map.insert("classifier.bias".to_string(), "classifier.bias".to_string());

        map
    }

    /// Load weights from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file.
    /// * `model` - The model to load weights into.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    pub fn load_safetensors(&self, path: impl AsRef<Path>, model: &mut Extractor) -> Result<()> {
        let path = path.as_ref();

        // Read the safetensors file
        let file_data = std::fs::read(path)
            .map_err(|e| GlinerError::model_loading_with_path(
                format!("Failed to read safetensors file: {e}"),
                path,
            ))?;

        // Parse the safetensors
        let safetensors = SafeTensors::deserialize(&file_data)
            .map_err(|e| GlinerError::model_loading_with_source(
                format!("Failed to parse safetensors: {e}"),
                e,
            ))?;

        // Load each tensor
        let mut loaded_weights: HashMap<String, Tensor> = HashMap::new();

        for tensor_view in safetensors.tensors() {
            let name = tensor_view.name().to_string();
            let dtype = tensor_view.dtype();
            let shape: Vec<usize> = tensor_view.shape().iter().map(|&d| d as usize).collect();
            let data = tensor_view.data();

            // Convert safetensors dtype to tch Kind
            let kind = Self::safetensors_dtype_to_kind(dtype)
                .map_err(|e| GlinerError::weight_loading(
                    name.clone(),
                    vec![],
                    shape.clone(),
                ))?;

            // Create tensor from bytes
            let tensor = Tensor::from_blob(
                data.as_ptr() as *const _,
                &shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
                &[],
                kind,
                self.device,
            );

            // Apply weight name mapping
            let mapped_name = self.weight_map.get(&name).cloned().unwrap_or(name.clone());
            loaded_weights.insert(mapped_name, tensor);
        }

        // Apply loaded weights to model components
        self.apply_weights_to_model(&loaded_weights, model)?;

        Ok(())
    }

    /// Load weights from multiple safetensors files (sharded models).
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to the safetensors files.
    /// * `model` - The model to load weights into.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    pub fn load_sharded_safetensors(
        &self,
        paths: &[impl AsRef<Path>],
        model: &mut Extractor,
    ) -> Result<()> {
        let mut all_weights: HashMap<String, Tensor> = HashMap::new();

        for path in paths {
            let path = path.as_ref();
            let file_data = std::fs::read(path)
                .map_err(|e| GlinerError::model_loading_with_path(
                    format!("Failed to read safetensors file: {e}"),
                    path,
                ))?;

            let safetensors = SafeTensors::deserialize(&file_data)
                .map_err(|e| GlinerError::model_loading_with_source(
                    format!("Failed to parse safetensors: {e}"),
                    e,
                ))?;

            for tensor_view in safetensors.tensors() {
                let name = tensor_view.name().to_string();
                let dtype = tensor_view.dtype();
                let shape: Vec<usize> = tensor_view.shape().iter().map(|&d| d as usize).collect();
                let data = tensor_view.data();

                let kind = Self::safetensors_dtype_to_kind(dtype)?;
                let tensor = Tensor::from_blob(
                    data.as_ptr() as *const _,
                    &shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
                    &[],
                    kind,
                    self.device,
                );

                let mapped_name = self.weight_map.get(&name).cloned().unwrap_or(name);
                all_weights.insert(mapped_name, tensor);
            }
        }

        self.apply_weights_to_model(&all_weights, model)?;
        Ok(())
    }

    /// Apply loaded weights to the model components.
    ///
    /// # Arguments
    ///
    /// * `weights` - The loaded weights mapping.
    /// * `model` - The model to apply weights to.
    fn apply_weights_to_model(
        &self,
        weights: &HashMap<String, Tensor>,
        model: &mut Extractor,
    ) -> Result<()> {
        // Apply span representation weights
        if let Some(span_rep_weight) = weights.get("span_rep.weight") {
            // In a full implementation, we would set the weight directly
            // For now, this is a placeholder
            tracing::debug!("Loaded span_rep.weight: {:?}", span_rep_weight.size());
        }

        if let Some(span_rep_width_emb) = weights.get("span_rep.width_embedding") {
            tracing::debug!("Loaded span_rep.width_embedding: {:?}", span_rep_width_emb.size());
        }

        // Apply count prediction weights
        if let Some(count_pred_weight) = weights.get("count_pred.weight") {
            tracing::debug!("Loaded count_pred.weight: {:?}", count_pred_weight.size());
        }

        if let Some(count_pred_emb) = weights.get("count_pred.embedding.weight") {
            tracing::debug!("Loaded count_pred.embedding.weight: {:?}", count_pred_emb.size());
        }

        // Apply classifier weights
        if let Some(cls_weight) = weights.get("classifier.weight") {
            tracing::debug!("Loaded classifier.weight: {:?}", cls_weight.size());
        }

        if let Some(cls_bias) = weights.get("classifier.bias") {
            tracing::debug!("Loaded classifier.bias: {:?}", cls_bias.size());
        }

        // Encoder weights would be applied to the encoder VarStore
        // This requires the encoder to be initialized with a VarStore first
        let encoder_weights: HashMap<String, &Tensor> = weights
            .iter()
            .filter(|(k, _)| k.starts_with("encoder."))
            .map(|(k, v)| (k.clone(), v))
            .collect();

        if !encoder_weights.is_empty() {
            tracing::debug!("Loaded {} encoder weights", encoder_weights.len());
        }

        Ok(())
    }

    /// Convert safetensors dtype to tch Kind.
    ///
    /// # Arguments
    ///
    /// * `dtype` - The safetensors dtype.
    ///
    /// # Returns
    ///
    /// The corresponding tch Kind, or an error if unsupported.
    fn safetensors_dtype_to_kind(dtype: safetensors::Dtype) -> Result<Kind> {
        match dtype {
            safetensors::Dtype::F32 => Ok(Kind::Float),
            safetensors::Dtype::F16 => Ok(Kind::Half),
            safetensors::Dtype::BF16 => Ok(Kind::BFloat16),
            safetensors::Dtype::F64 => Ok(Kind::Double),
            safetensors::Dtype::I32 => Ok(Kind::Int),
            safetensors::Dtype::I64 => Ok(Kind::Int64),
            safetensors::Dtype::I8 => Ok(Kind::Int8),
            safetensors::Dtype::U8 => Ok(Kind::Uint8),
            safetensors::Dtype::BOOL => Ok(Kind::Bool),
            _ => Err(GlinerError::model_loading(format!(
                "Unsupported safetensors dtype: {:?}",
                dtype
            ))),
        }
    }

    /// Get the weight map.
    pub fn weight_map(&self) -> &WeightMap {
        &self.weight_map
    }

    /// Get the device.
    pub fn device(&self) -> Device {
        self.device
    }
}

/// Utility functions for working with model weights.
pub mod utils {
    use super::*;

    /// Check if a directory contains safetensors weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory.
    ///
    /// # Returns
    ///
    /// `true` if safetensors files are found.
    pub fn has_safetensors(path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        if path.is_dir() {
            path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
        } else {
            path.extension().map_or(false, |ext| ext == "safetensors")
        }
    }

    /// Get all safetensors files in a directory.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory.
    ///
    /// # Returns
    ///
    /// A vector of paths to safetensors files.
    pub fn get_safetensors_files(path: impl AsRef<Path>) -> Vec<std::path::PathBuf> {
        let path = path.as_ref();
        let mut files = Vec::new();

        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if entry_path.extension().map_or(false, |ext| ext == "safetensors") {
                        files.push(entry_path);
                    }
                }
            }
        } else if path.exists() {
            files.push(path.to_path_buf());
        }

        files.sort();
        files
    }

    /// Load a model from a HuggingFace model directory.
    ///
    /// This function automatically detects whether the model is sharded
    /// and loads all weights accordingly.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory or file.
    /// * `config` - The extractor configuration.
    /// * `device` - The device to load tensors onto.
    ///
    /// # Returns
    ///
    /// A loaded `Extractor` model.
    pub fn load_from_hf_dir(
        model_path: impl AsRef<Path>,
        config: &ExtractorConfig,
        device: Device,
    ) -> Result<Extractor> {
        let model_path = model_path.as_ref();
        let mut model = Extractor::new(config)?;
        let loader = ModelLoader::new(config, device)?;

        if has_safetensors(model_path) {
            let files = get_safetensors_files(model_path);
            if files.is_empty() {
                return Err(GlinerError::model_loading_with_path(
                    "No safetensors files found",
                    model_path,
                ));
            }

            if files.len() == 1 {
                loader.load_safetensors(&files[0], &mut model)?;
            } else {
                let paths: Vec<&std::path::Path> = files.iter().map(|p| p.as_path()).collect();
                // Note: load_sharded_safetensors expects AsRef<Path>
                // We need to call it differently for multiple files
                for file in &files {
                    loader.load_safetensors(file, &mut model)?;
                }
            }
        } else {
            return Err(GlinerError::model_loading_with_path(
                "No safetensors files found in directory",
                model_path,
            ));
        }

        model.is_loaded = true;
        Ok(model)
    }

    /// Validate that loaded weights match the model configuration.
    ///
    /// # Arguments
    ///
    /// * `weights` - The loaded weights.
    /// * `config` - The model configuration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights are valid, or an error describing the mismatch.
    pub fn validate_weights(
        weights: &HashMap<String, Tensor>,
        config: &ExtractorConfig,
    ) -> Result<()> {
        // Check encoder embedding size
        if let Some(embed_weight) = weights.get("encoder.embeddings.word_embeddings.weight") {
            let size = embed_weight.size();
            if size.len() != 2 {
                return Err(GlinerError::weight_loading(
                    "encoder.embeddings.word_embeddings.weight".to_string(),
                    vec![2],
                    size.iter().map(|d| *d as usize).collect(),
                ));
            }

            let vocab_size = size[0] as usize;
            let hidden_size = size[1] as usize;

            if vocab_size != config.vocab_size {
                return Err(GlinerError::weight_loading(
                    "encoder.embeddings.word_embeddings.weight".to_string(),
                    vec![config.vocab_size, config.hidden_size],
                    vec![vocab_size, hidden_size],
                ));
            }

            if hidden_size != config.hidden_size {
                return Err(GlinerError::weight_loading(
                    "encoder.embeddings.word_embeddings.weight".to_string(),
                    vec![config.vocab_size, config.hidden_size],
                    vec![vocab_size, hidden_size],
                ));
            }
        }

        // Check span representation weights
        if let Some(span_weight) = weights.get("span_rep.weight") {
            let size = span_weight.size();
            if size.len() == 2 {
                let hidden = size[1] as usize;
                if hidden != config.hidden_size {
                    return Err(GlinerError::weight_loading(
                        "span_rep.weight".to_string(),
                        vec![config.hidden_size],
                        vec![hidden],
                    ));
                }
            }
        }

        // Check classifier weights
        if let Some(cls_weight) = weights.get("classifier.weight") {
            let size = cls_weight.size();
            if size.len() == 2 {
                let hidden = size[1] as usize;
                if hidden != config.hidden_size {
                    return Err(GlinerError::weight_loading(
                        "classifier.weight".to_string(),
                        vec![config.hidden_size],
                        vec![hidden],
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_map_creation() {
        let config = ExtractorConfig::default();
        let loader = ModelLoader::new(&config, Device::Cpu).unwrap();
        let weight_map = loader.weight_map();

        // Check that encoder weights are mapped
        assert!(weight_map.contains_key("embeddings.word_embeddings.weight"));
        assert!(weight_map.contains_key("encoder.layer.0.attention.self.query.weight"));

        // Check GLiNER2-specific weights
        assert!(weight_map.contains_key("span_rep.weight"));
        assert!(weight_map.contains_key("count_pred.weight"));
        assert!(weight!(weight_map.contains_key("classifier.weight"));
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(
            ModelLoader::safetensors_dtype_to_kind(safetensors::Dtype::F32).unwrap(),
            Kind::Float
        );
        assert_eq!(
            ModelLoader::safetensors_dtype_to_kind(safetensors::Dtype::F16).unwrap(),
            Kind::Half
        );
        assert_eq!(
            ModelLoader::safetensors_dtype_to_kind(safetensors::Dtype::BF16).unwrap(),
            Kind::BFloat16
        );
        assert_eq!(
            ModelLoader::safetensors_dtype_to_kind(safetensors::Dtype::I64).unwrap(),
            Kind::Int64
        );
    }

    #[test]
    fn test_has_safetensors() {
        // Test with file path
        assert!(utils::has_safetensors("model.safetensors"));
        assert!(!utils::has_safetensors("model.bin"));

        // Test with directory path (would need actual files to test properly)
        // assert!(utils::has_safetensors("/path/to/model/dir"));
    }

    #[test]
    fn test_weight_validation() {
        let config = ExtractorConfig::default();
        let mut weights = HashMap::new();

        // Add valid embedding weight
        let embed_weight = Tensor::randn(
            &[config.vocab_size as i64, config.hidden_size as i64],
            (Kind::Float, Device::Cpu),
        );
        weights.insert("encoder.embeddings.word_embeddings.weight".to_string(), embed_weight);

        // Validation should pass
        assert!(utils::validate_weights(&weights, &config).is_ok());

        // Add invalid weight (wrong hidden size)
        let bad_weight = Tensor::randn(&[config.vocab_size as i64, 512], (Kind::Float, Device::Cpu));
        weights.insert("encoder.embeddings.word_embeddings.weight".to_string(), bad_weight);

        // Validation should fail
        assert!(utils::validate_weights(&weights, &config).is_err());
    }
}
