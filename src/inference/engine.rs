//! Inference engine for GLiNER2.
//!
//! This module provides the main `GLiNER2` struct that serves as the
//! user-facing API for information extraction. It coordinates model loading,
//! schema processing, batch inference, and result formatting.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::GLiNER2;
//! use gliner2_rust::schema::SchemaBuilder;
//!
//! // Load model
//! let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
//!
//! // Build schema
//! let schema = SchemaBuilder::new()
//!     .entity("person").description("Names of people").done()
//!     .entity("company").done()
//!     .build()?;
//!
//! // Extract entities
//! let result = model.extract_entities("Apple CEO Tim Cook", &schema)?;
//! println!("{:?}", result);
//! ```

use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;
use serde_json::Value as JsonValue;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer as HfTokenizer;

use crate::batch::{ExtractorCollator, PreprocessedBatch};
use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};
use crate::model::Extractor;
use crate::schema::builder::SchemaBuilder;
use crate::schema::types::{EntityDef, FieldDtype, RegexValidator, Schema, TaskType};
use crate::tokenizer::WhitespaceTokenizer;

/// Extraction result for a single text.
pub type ExtractionResult = JsonValue;

/// Batch extraction results for multiple texts.
pub type BatchExtractionResult = Vec<ExtractionResult>;

/// Main GLiNER2 inference engine.
///
/// This struct provides the primary API for running GLiNER2 inference.
/// It handles model loading, schema processing, batch inference, and
/// result formatting.
///
/// # Thread Safety
///
/// `GLiNER2` is `Send + Sync` and can be shared across threads using `Arc`.
///
/// # Example
///
/// ```ignore
/// use gliner2_rust::GLiNER2;
///
/// let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
/// let result = model.extract_entities(
///     "Apple CEO Tim Cook announced iPhone 15.",
///     &["company", "person", "product"]
/// )?;
/// ```
#[derive(Debug)]
pub struct GLiNER2 {
    /// The underlying extractor model.
    model: Extractor,
    /// Whitespace tokenizer for text preprocessing.
    tokenizer: WhitespaceTokenizer,
    /// Batch collator for inference.
    collator: ExtractorCollator,
    /// Default confidence threshold.
    default_threshold: f32,
    /// Device for inference.
    device: Device,
}

impl GLiNER2 {
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// Create a new GLiNER2 engine from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    ///
    /// # Returns
    ///
    /// A new `GLiNER2` engine.
    pub fn new(config: &ExtractorConfig) -> Result<Self> {
        let model = Extractor::new(config)?;
        let ws_tokenizer = WhitespaceTokenizer::new();

        // Try to load HuggingFace tokenizer (default behavior)
        let hf_tokenizer = Self::load_hf_tokenizer(config);
        let collator = if let Some(hf_tok) = hf_tokenizer {
            ExtractorCollator::with_hf_tokenizer(
                ws_tokenizer.clone(),
                hf_tok,
                false,
                config.max_len,
            )
        } else {
            ExtractorCollator::with_max_len(
                ws_tokenizer.clone(),
                false,
                config.max_len,
            )
        };

        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };

        Ok(Self {
            model,
            tokenizer: ws_tokenizer,
            collator,
            default_threshold: 0.5,
            device,
        })
    }

    /// Load a GLiNER2 model from a pretrained repository or local path.
    ///
    /// # Arguments
    ///
    /// * `model_name_or_path` - HuggingFace model name or local path.
    ///
    /// # Returns
    ///
    /// A loaded `GLiNER2` engine.
    pub fn from_pretrained(model_name_or_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_name_or_path.as_ref();
        let config = ExtractorConfig::new(
            path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "fastino/gliner2-base-v1".to_string()),
        );

        let mut model = Extractor::new(&config)?;

        // Try to load weights if path exists
        if path.exists() {
            model.load_weights(path)?;
        }

        let ws_tokenizer = WhitespaceTokenizer::new();

        // Try to load HuggingFace tokenizer from local path or model name
        let hf_tokenizer = if path.exists() {
            // Try loading from local directory first
            Self::load_hf_tokenizer_from_path(path).or_else(|| Self::load_hf_tokenizer(&config))
        } else {
            Self::load_hf_tokenizer(&config)
        };

        let collator = if let Some(hf_tok) = hf_tokenizer {
            ExtractorCollator::with_hf_tokenizer(
                ws_tokenizer.clone(),
                hf_tok,
                false,
                config.max_len,
            )
        } else {
            ExtractorCollator::with_max_len(
                ws_tokenizer.clone(),
                false,
                config.max_len,
            )
        };

        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };

        Ok(Self {
            model,
            tokenizer: ws_tokenizer,
            collator,
            default_threshold: 0.5,
            device,
        })
    }

    // -------------------------------------------------------------------------
    // Tokenizer Loading
    // -------------------------------------------------------------------------

    /// Load a HuggingFace tokenizer from config.
    ///
    /// Tries to load from `config.tokenizer_path` first, then from local model directory,
    /// then downloads from HuggingFace Hub using the model name.
    ///
    /// Returns `None` only if all loading methods fail (falls back to whitespace tokenizer).
    fn load_hf_tokenizer(config: &ExtractorConfig) -> Option<HfTokenizer> {
        // Try loading from explicit tokenizer path first
        if let Some(ref path) = config.tokenizer_path {
            if let Some(tokenizer) = Self::load_hf_tokenizer_from_path(path) {
                tracing::info!("Loaded HuggingFace tokenizer from: {:?}", path);
                return Some(tokenizer);
            }
        }

        // Try loading from model name as local directory
        let model_name = &config.model_name;
        let model_path = std::path::Path::new(model_name);
        if model_path.exists() && model_path.is_dir() {
            if let Some(tokenizer) = Self::load_hf_tokenizer_from_path(model_path) {
                tracing::info!("Loaded HuggingFace tokenizer from model directory: {}", model_name);
                return Some(tokenizer);
            }
        }

        // Try common tokenizer file patterns in current directory
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
        for file in &tokenizer_files {
            let path = std::path::Path::new(file);
            if path.exists() {
                if let Ok(tokenizer) = HfTokenizer::from_file(path) {
                    tracing::info!("Loaded HuggingFace tokenizer from: {}", file);
                    return Some(tokenizer);
                }
            }
        }

        // Download from HuggingFace Hub (default behavior)
        tracing::info!("Downloading tokenizer for '{}' from HuggingFace Hub...", model_name);
        Self::download_hf_tokenizer(model_name)
    }

    /// Load a HuggingFace tokenizer from a directory path.
    ///
    /// Looks for `tokenizer.json` first (fast), then falls back to
    /// loading from the path directly if it's a JSON file.
    fn load_hf_tokenizer_from_path(path: impl AsRef<std::path::Path>) -> Option<HfTokenizer> {
        let path = path.as_ref();

        // Try loading tokenizer.json first (preferred, fast)
        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            if let Ok(tokenizer) = HfTokenizer::from_file(&tokenizer_json) {
                return Some(tokenizer);
            }
        }

        // Try loading from the path directly if it's a tokenizer.json file
        if path.extension().map_or(false, |ext| ext == "json") {
            if let Ok(tokenizer) = HfTokenizer::from_file(path) {
                return Some(tokenizer);
            }
        }

        None
    }

    /// Download a HuggingFace tokenizer from the Hub.
    ///
    /// Downloads `tokenizer.json` for the given model ID and loads it.
    /// The file is cached locally by `hf-hub` for future use.
    fn download_hf_tokenizer(model_id: &str) -> Option<HfTokenizer> {
        use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        );

        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .ok()?;

        let repo_api = api.repo(repo);

        // Download tokenizer.json
        match repo_api.get("tokenizer.json") {
            Ok(tokenizer_path) => {
                tracing::info!("Downloaded tokenizer.json to: {:?}", tokenizer_path);
                HfTokenizer::from_file(&tokenizer_path).ok()
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to download tokenizer for '{}': {}. \
                     Falling back to whitespace tokenizer.",
                    model_id,
                    e
                );
                None
            }
        }
    }

    // -------------------------------------------------------------------------
    // Schema & Extraction
    // -------------------------------------------------------------------------

    /// Create a new schema builder.
    ///
    /// # Returns
    ///
    /// A new `SchemaBuilder` for constructing extraction schemas.
    pub fn create_schema(&self) -> SchemaBuilder {
        SchemaBuilder::new()
    }

    // -------------------------------------------------------------------------
    // Entity Extraction
    // -------------------------------------------------------------------------

    /// Extract entities from a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `entity_types` - Entity types to extract (names or with descriptions).
    /// * `threshold` - Confidence threshold (default: 0.5).
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length (overrides config).
    ///
    /// # Returns
    ///
    /// Extraction result with entities.
    pub fn extract_entities(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: Option<f32>,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<ExtractionResult> {
        let schema = self.build_entity_schema(entity_types)?;
        self.extract(
            text,
            &schema,
            threshold.unwrap_or(self.default_threshold),
            include_confidence,
            include_spans,
            max_len,
        )
    }

    /// Batch extract entities from multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `entity_types` - Entity types to extract.
    /// * `batch_size` - Batch size for processing.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Batch extraction results.
    pub fn batch_extract_entities(
        &self,
        texts: &[String],
        entity_types: &[&str],
        batch_size: usize,
        threshold: Option<f32>,
        num_workers: usize,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<BatchExtractionResult> {
        let schema = self.build_entity_schema(entity_types)?;
        self.batch_extract(
            texts,
            &schema,
            batch_size,
            threshold.unwrap_or(self.default_threshold),
            num_workers,
            include_confidence,
            include_spans,
            max_len,
        )
    }

    // -------------------------------------------------------------------------
    // Classification
    // -------------------------------------------------------------------------

    /// Classify a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `tasks` - Classification tasks (task name -> labels).
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    ///
    /// # Returns
    ///
    /// Classification result.
    pub fn classify_text(
        &self,
        text: &str,
        tasks: &[(String, Vec<String>)],
        threshold: Option<f32>,
        include_confidence: bool,
    ) -> Result<ExtractionResult> {
        let mut schema_builder = self.create_schema();
        for (task_name, labels) in tasks {
            schema_builder = schema_builder
                .classification(task_name.clone(), labels.clone())
                .threshold(threshold.unwrap_or(0.5))
                .done();
        }
        let schema = schema_builder.build()?;

        self.extract(
            text,
            &schema,
            threshold.unwrap_or(self.default_threshold),
            include_confidence,
            false,
            None,
        )
    }

    /// Batch classify multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `tasks` - Classification tasks.
    /// * `batch_size` - Batch size.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers.
    /// * `include_confidence` - Whether to include confidence scores.
    ///
    /// # Returns
    ///
    /// Batch classification results.
    pub fn batch_classify_text(
        &self,
        texts: &[String],
        tasks: &[(String, Vec<String>)],
        batch_size: usize,
        threshold: Option<f32>,
        num_workers: usize,
        include_confidence: bool,
    ) -> Result<BatchExtractionResult> {
        let mut schema_builder = self.create_schema();
        for (task_name, labels) in tasks {
            schema_builder = schema_builder
                .classification(task_name.clone(), labels.clone())
                .threshold(threshold.unwrap_or(0.5))
                .done();
        }
        let schema = schema_builder.build()?;

        self.batch_extract(
            texts,
            &schema,
            batch_size,
            threshold.unwrap_or(self.default_threshold),
            num_workers,
            include_confidence,
            false,
            None,
        )
    }

    // -------------------------------------------------------------------------
    // Relation Extraction
    // -------------------------------------------------------------------------

    /// Extract relations from a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `relation_types` - Relation types to extract.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    ///
    /// # Returns
    ///
    /// Extraction result with relations.
    pub fn extract_relations(
        &self,
        text: &str,
        relation_types: &[&str],
        threshold: Option<f32>,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<ExtractionResult> {
        let mut schema_builder = self.create_schema();
        for rel_type in relation_types {
            schema_builder = schema_builder.relation(rel_type.to_string()).done();
        }
        let schema = schema_builder.build()?;

        self.extract(
            text,
            &schema,
            threshold.unwrap_or(self.default_threshold),
            include_confidence,
            include_spans,
            None,
        )
    }

    /// Batch extract relations from multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `relation_types` - Relation types to extract.
    /// * `batch_size` - Batch size.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    ///
    /// # Returns
    ///
    /// Batch extraction results.
    pub fn batch_extract_relations(
        &self,
        texts: &[String],
        relation_types: &[&str],
        batch_size: usize,
        threshold: Option<f32>,
        num_workers: usize,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<BatchExtractionResult> {
        let mut schema_builder = self.create_schema();
        for rel_type in relation_types {
            schema_builder = schema_builder.relation(rel_type.to_string()).done();
        }
        let schema = schema_builder.build()?;

        self.batch_extract(
            texts,
            &schema,
            batch_size,
            threshold.unwrap_or(self.default_threshold),
            num_workers,
            include_confidence,
            include_spans,
            None,
        )
    }

    // -------------------------------------------------------------------------
    // General Extraction
    // -------------------------------------------------------------------------

    /// Extract from a single text using a custom schema.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `schema` - The extraction schema.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Extraction result.
    pub fn extract(
        &self,
        text: &str,
        schema: &Schema,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<ExtractionResult> {
        let results = self.batch_extract(
            &[text.to_string()],
            schema,
            1,
            threshold,
            0,
            include_confidence,
            include_spans,
            max_len,
        )?;

        results.into_iter().next().ok_or_else(|| {
            GlinerError::inference("No results returned from extraction")
        })
    }

    /// Batch extract from multiple texts using a schema.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `schema` - The extraction schema.
    /// * `batch_size` - Batch size for processing.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers for preprocessing.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Batch extraction results.
    pub fn batch_extract(
        &self,
        texts: &[String],
        schema: &Schema,
        batch_size: usize,
        threshold: f32,
        num_workers: usize,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<BatchExtractionResult> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Set model to evaluation mode
        let model = &self.model;
        // Note: eval() requires mutable access, but we use immutable reference here
        // The is_training flag is only used internally and doesn't affect inference

        // Convert schema to dict format
        let schema_dict = schema.to_dict();

        // Create samples
        let samples: Vec<(String, JsonValue)> = texts
            .iter()
            .map(|text| (text.clone(), schema_dict.clone()))
            .collect();

        // Process in batches
        let mut all_results = Vec::with_capacity(texts.len());

        // Sequential processing (Tensor is not Sync in tch 0.24)
        let chunks: Vec<_> = samples.chunks(batch_size).collect();
        for chunk in chunks {
            let results = self.process_batch(
                chunk,
                model,
                threshold,
                include_confidence,
                include_spans,
                max_len,
            )?;
            all_results.extend(results);
        }

        Ok(all_results)
    }

    // -------------------------------------------------------------------------
    // Internal Methods
    // -------------------------------------------------------------------------

    /// Process a single batch of samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Batch of (text, schema) pairs.
    /// * `model` - The extractor model.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Extraction results for the batch.
    fn process_batch(
        &self,
        samples: &[(String, JsonValue)],
        model: &Extractor,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<Vec<ExtractionResult>> {
        // Collate batch
        let batch = self.collator.collate(samples)?;

        // Move to device
        let batch = batch.to(self.device.clone(), None)?;

        // Run forward pass
        let output = model.forward(&batch)?;

        // Extract results for each sample
        let mut results = Vec::with_capacity(batch.batch_size());

        for sample_idx in 0..batch.batch_size() {
            let result = self.extract_sample(
                &output,
                &batch,
                sample_idx,
                threshold,
                include_confidence,
                include_spans,
            )?;
            results.push(result);
        }

        Ok(results)
    }

    /// Extract results for a single sample from model output.
    ///
    /// # Arguments
    ///
    /// * `output` - Model output.
    /// * `batch` - Preprocessed batch.
    /// * `sample_idx` - Sample index.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    ///
    /// # Returns
    ///
    /// Extraction result for the sample.
    fn extract_sample(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<ExtractionResult> {
        let mut result = serde_json::Map::new();

        // Get task types for this sample
        let task_types = batch.sample_task_types(sample_idx).unwrap_or(&[]);

        // Process each schema/task
        for (schema_idx, task_type) in task_types.iter().enumerate() {
            let task_result = match task_type.as_str() {
                "entities" => self.extract_entities_from_output(
                    output, batch, sample_idx, schema_idx, threshold,
                    include_confidence, include_spans,
                )?,
                "classifications" => self.extract_classification_from_output(
                    output, batch, sample_idx, schema_idx, threshold,
                    include_confidence,
                )?,
                "relations" => self.extract_relations_from_output(
                    output, batch, sample_idx, schema_idx, threshold,
                    include_confidence, include_spans,
                )?,
                "json_structures" => self.extract_structures_from_output(
                    output, batch, sample_idx, schema_idx, threshold,
                    include_confidence, include_spans,
                )?,
                _ => JsonValue::Null,
            };

            // Add to result based on task type
            match task_type.as_str() {
                "entities" => {
                    result.insert("entities".to_string(), task_result);
                }
                "classifications" => {
                    // Get task name from schema tokens
                    if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx) {
                        if schema_tokens.len() > 2 {
                            let task_name = &schema_tokens[2];
                            result.insert(task_name.clone(), task_result);
                        }
                    }
                }
                "relations" => {
                    if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx) {
                        if schema_tokens.len() > 2 {
                            let rel_name = &schema_tokens[2];
                            if let Some(relations) = result.get_mut("relation_extraction") {
                                if let Some(rel_obj) = relations.as_object_mut() {
                                    rel_obj.insert(rel_name.clone(), task_result);
                                }
                            } else {
                                let mut rel_obj = serde_json::Map::new();
                                rel_obj.insert(schema_tokens[2].clone(), task_result);
                                result.insert("relation_extraction".to_string(), JsonValue::Object(rel_obj));
                            }
                        }
                    }
                }
                "json_structures" => {
                    if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx) {
                        if schema_tokens.len() > 2 {
                            let struct_name = &schema_tokens[2];
                            result.insert(struct_name.clone(), task_result);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(JsonValue::Object(result))
    }

    /// Extract entities from model output.
    fn extract_entities_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonValue> {
        // Placeholder implementation - in full version, this would:
        // 1. Get span representations
        // 2. Compute span scores
        // 3. Apply threshold filtering
        // 4. Format results

        let mut entities = serde_json::Map::new();

        // Get schema tokens to identify entity types
        if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx) {
            for token in schema_tokens {
                if token.starts_with("[E]") {
                    // This is an entity type
                    let entity_type = token.trim_start_matches("[E] ").trim_start_matches("[E]");
                    entities.insert(
                        entity_type.to_string(),
                        JsonValue::Array(Vec::new()), // Placeholder
                    );
                }
            }
        }

        Ok(JsonValue::Object(entities))
    }

    /// Extract classification from model output.
    fn extract_classification_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
    ) -> Result<JsonValue> {
        // Placeholder implementation
        Ok(JsonValue::String("unknown".to_string()))
    }

    /// Extract relations from model output.
    fn extract_relations_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonValue> {
        // Placeholder implementation
        Ok(JsonValue::Array(Vec::new()))
    }

    /// Extract structures from model output.
    fn extract_structures_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonValue> {
        // Placeholder implementation
        Ok(JsonValue::Array(Vec::new()))
    }

    // -------------------------------------------------------------------------
    // Helper Methods
    // -------------------------------------------------------------------------

    /// Build an entity schema from entity type names.
    ///
    /// # Arguments
    ///
    /// * `entity_types` - Entity type names.
    ///
    /// # Returns
    ///
    /// A constructed schema.
    fn build_entity_schema(&self, entity_types: &[&str]) -> Result<Schema> {
        let mut builder = self.create_schema();
        for entity_type in entity_types {
            builder = builder.entity(entity_type.to_string()).done();
        }
        builder.build()
    }

    /// Apply regex validators to extracted text.
    ///
    /// # Arguments
    ///
    /// * `text` - The extracted text.
    /// * `validators` - Regex validators to apply.
    ///
    /// # Returns
    ///
    /// `true` if the text passes all validators.
    fn apply_validators(&self, text: &str, validators: &[RegexValidator]) -> bool {
        if validators.is_empty() {
            return true;
        }

        validators.iter().all(|v| v.validate(text).unwrap_or(false))
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Get the model configuration.
    pub fn config(&self) -> &ExtractorConfig {
        &self.model.config
    }

    /// Get the default threshold.
    pub fn default_threshold(&self) -> f32 {
        self.default_threshold
    }

    /// Set the default threshold.
    pub fn set_default_threshold(&mut self, threshold: f32) {
        self.default_threshold = threshold;
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Quantize the model to FP16.
    pub fn quantize(&mut self) -> Result<()> {
        self.model.quantize()
    }

    /// Compile the model for faster inference.
    pub fn compile(&mut self) -> Result<()> {
        self.model.compile()
    }

    /// Load model weights from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    pub fn load_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.model.load_weights(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gliner2_creation() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config);
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        assert_eq!(engine.default_threshold(), 0.5);
    }

    #[test]
    fn test_schema_builder() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config).unwrap();
        let schema_builder = engine.create_schema();
        assert!(schema_builder.build().is_err()); // Empty schema is invalid

        let schema = engine.create_schema()
            .entity("person").done()
            .entity("company").done()
            .build();
        assert!(schema.is_ok());
    }

    #[test]
    fn test_build_entity_schema() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config).unwrap();
        let schema = engine.build_entity_schema(&["person", "company"]);
        assert!(schema.is_ok());
        let schema = schema.unwrap();
        assert_eq!(schema.entities.len(), 2);
    }

    #[test]
    fn test_apply_validators() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config).unwrap();

        // No validators - should pass
        assert!(engine.apply_validators("test", &[]));

        // With valid regex
        let validator = RegexValidator::new(r"^\d+$").unwrap();
        assert!(engine.apply_validators("123", &[validator.clone()]));
        assert!(!engine.apply_validators("abc", &[validator]));
    }

    #[test]
    fn test_set_threshold() {
        let config = ExtractorConfig::default();
        let mut engine = GLiNER2::new(&config).unwrap();
        assert_eq!(engine.default_threshold(), 0.5);
        engine.set_default_threshold(0.7);
        assert_eq!(engine.default_threshold(), 0.7);
    }
}
