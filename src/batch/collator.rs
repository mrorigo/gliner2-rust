//! ExtractorCollator for GLiNER2 batch preprocessing.
//!
//! This module provides the `ExtractorCollator` struct that converts raw
//! text/schema pairs into GPU-ready `PreprocessedBatch` objects. It handles
//! tokenization, schema encoding, padding, and index mapping for efficient
//! batch inference.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::batch::ExtractorCollator;
//! use gliner2_rust::schema::Schema;
//! use gliner2_rust::tokenizer::WhitespaceTokenizer;
//!
//! let tokenizer = WhitespaceTokenizer::new();
//! let collator = ExtractorCollator::new(&tokenizer, false);
//!
//! let samples = vec![
//!     ("Apple CEO Tim Cook", schema_dict),
//!     ("Google in Mountain View", schema_dict),
//! ];
//!
//! let batch = collator.collate(&samples)?;
//! ```

use serde_json::Value as JsonValue;
use tch::Tensor;

use crate::batch::preprocessed::{PreprocessedBatch, TokenMapping};
use crate::error::{GlinerError, Result};
use crate::schema::types::{Schema, TaskType};
use crate::tokenizer::WhitespaceTokenizer;

/// Special tokens used in GLiNER2 schema encoding.
pub mod special_tokens {
    /// Opening parenthesis.
    pub const OPEN_PAREN: &str = "(";
    /// Closing parenthesis.
    pub const CLOSE_PAREN: &str = ")";
    /// Prompt token.
    pub const P_TOKEN: &str = "[P]";
    /// Entity token.
    pub const E_TOKEN: &str = "[E]";
    /// Classification token.
    pub const L_TOKEN: &str = "[C]";
    /// Relation token.
    pub const R_TOKEN: &str = "[R]";
    /// Description token.
    pub const DESC_TOKEN: &str = "[DESCRIPTION]";
    /// Example token.
    pub const EXAMPLE_TOKEN: &str = "[EXAMPLE]";
    /// Output token.
    pub const OUTPUT_TOKEN: &str = "[OUTPUT]";
}

/// A single sample for collation: (text, schema).
pub type CollateSample = (String, JsonValue);

/// ExtractorCollator for converting text/schema pairs into batches.
///
/// This collator handles all preprocessing for GLiNER2 inference,
/// including:
/// - Whitespace tokenization of input text
/// - Schema encoding into token sequences
/// - Padding and batching
/// - Index mapping for embedding extraction
///
/// # Arguments
///
/// * `tokenizer` - The whitespace tokenizer to use.
/// * `is_training` - Whether in training mode (affects schema processing).
/// * `max_len` - Optional maximum token length for truncation.
#[derive(Debug, Clone)]
pub struct ExtractorCollator {
    /// The whitespace tokenizer.
    tokenizer: WhitespaceTokenizer,
    /// Whether in training mode.
    is_training: bool,
    /// Maximum token length (None = no limit).
    max_len: Option<usize>,
}

impl ExtractorCollator {
    /// Create a new collator.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - The whitespace tokenizer.
    /// * `is_training` - Whether in training mode.
    pub fn new(tokenizer: WhitespaceTokenizer, is_training: bool) -> Self {
        Self {
            tokenizer,
            is_training,
            max_len: None,
        }
    }

    /// Create a new collator with max_len truncation.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - The whitespace tokenizer.
    /// * `is_training` - Whether in training mode.
    /// * `max_len` - Maximum token length.
    pub fn with_max_len(
        tokenizer: WhitespaceTokenizer,
        is_training: bool,
        max_len: Option<usize>,
    ) -> Self {
        Self {
            tokenizer,
            is_training,
            max_len,
        }
    }

    /// Collate a batch of samples into a `PreprocessedBatch`.
    ///
    /// # Arguments
    ///
    /// * `samples` - A slice of (text, schema) pairs.
    ///
    /// # Returns
    ///
    /// A `PreprocessedBatch` ready for model inference.
    pub fn collate(&self, samples: &[CollateSample]) -> Result<PreprocessedBatch> {
        if samples.is_empty() {
            return Err(GlinerError::batch_processing(
                "Cannot collate empty batch",
            ));
        }

        // Process each sample
        let mut all_input_ids: Vec<Vec<i64>> = Vec::with_capacity(samples.len());
        let mut all_mapped_indices: Vec<Vec<TokenMapping>> = Vec::with_capacity(samples.len());
        let mut all_schema_counts: Vec<usize> = Vec::with_capacity(samples.len());
        let mut all_original_lengths: Vec<usize> = Vec::with_capacity(samples.len());
        let mut all_structure_labels: Vec<Vec<JsonValue>> = Vec::with_capacity(samples.len());
        let mut all_task_types: Vec<Vec<String>> = Vec::with_capacity(samples.len());
        let mut all_text_tokens: Vec<Vec<String>> = Vec::with_capacity(samples.len());
        let mut all_schema_tokens_list: Vec<Vec<Vec<String>>> = Vec::with_capacity(samples.len());
        let mut all_start_mappings: Vec<Vec<usize>> = Vec::with_capacity(samples.len());
        let mut all_end_mappings: Vec<Vec<usize>> = Vec::with_capacity(samples.len());
        let mut all_original_texts: Vec<String> = Vec::with_capacity(samples.len());
        let mut all_original_schemas: Vec<JsonValue> = Vec::with_capacity(samples.len());
        let mut all_text_word_indices: Vec<Vec<i64>> = Vec::with_capacity(samples.len());
        let mut all_text_word_counts: Vec<usize> = Vec::with_capacity(samples.len());
        let mut all_schema_special_indices: Vec<Vec<Vec<usize>>> = Vec::with_capacity(samples.len());

        for (text, schema) in samples {
            let processed = self.process_sample(text, schema)?;

            all_input_ids.push(processed.input_ids);
            all_mapped_indices.push(processed.mapped_indices);
            all_schema_counts.push(processed.schema_count);
            all_original_lengths.push(processed.original_length);
            all_structure_labels.push(processed.structure_labels);
            all_task_types.push(processed.task_types);
            all_text_tokens.push(processed.text_tokens);
            all_schema_tokens_list.push(processed.schema_tokens_list);
            all_start_mappings.push(processed.start_mapping);
            all_end_mappings.push(processed.end_mapping);
            all_original_texts.push(processed.original_text);
            all_original_schemas.push(processed.original_schema.clone());
            all_text_word_indices.push(processed.text_word_indices);
            all_text_word_counts.push(processed.text_word_count);
            all_schema_special_indices.push(processed.schema_special_indices);
        }

        // Find max sequence length for padding
        let max_seq_len = all_input_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

        // Pad input IDs and attention mask
        let mut padded_input_ids: Vec<i64> = Vec::with_capacity(samples.len() * max_seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(samples.len() * max_seq_len);

        for ids in &all_input_ids {
            let len = ids.len();
            padded_input_ids.extend_from_slice(ids);
            // Pad with 0 (assuming 0 is pad token ID)
            padded_input_ids.extend(std::iter::repeat(0).take(max_seq_len - len));

            // Attention mask: 1 for real tokens, 0 for padding
            attention_mask.extend(std::iter::repeat(1).take(len));
            attention_mask.extend(std::iter::repeat(0).take(max_seq_len - len));
        }

        // Create tensors
        let input_ids_tensor = Tensor::from_slice(&padded_input_ids)
            .view((samples.len() as i64, max_seq_len as i64))
            .to_kind(tch::Kind::Int64);

        let attention_mask_tensor = Tensor::from_slice(&attention_mask)
            .view((samples.len() as i64, max_seq_len as i64))
            .to_kind(tch::Kind::Int64);

        // Create text word indices tensor if all samples have the same max words
        let max_words = all_text_word_indices.iter().map(|idx| idx.len()).max().unwrap_or(0);
        let text_word_indices_tensor = if max_words > 0 {
            let mut flat_indices: Vec<i64> = Vec::with_capacity(samples.len() * max_words);
            for indices in &all_text_word_indices {
                flat_indices.extend_from_slice(indices);
                flat_indices.extend(std::iter::repeat(-1).take(max_words - indices.len()));
            }
            Some(
                Tensor::from_slice(&flat_indices)
                    .view((samples.len() as i64, max_words as i64))
                    .to_kind(tch::Kind::Int64),
            )
        } else {
            None
        };

        // Build the batch
        PreprocessedBatch::new(
            input_ids_tensor,
            attention_mask_tensor,
            all_mapped_indices,
            all_schema_counts,
            all_original_lengths,
            all_structure_labels,
            all_task_types,
            all_text_tokens,
            all_schema_tokens_list,
            all_start_mappings,
            all_end_mappings,
            all_original_texts,
            all_original_schemas,
            text_word_indices_tensor,
            all_text_word_counts,
            all_schema_special_indices,
        )
    }

    /// Process a single sample into tokenized form.
    fn process_sample(&self, text: &str, schema: &JsonValue) -> Result<ProcessedSample> {
        // Tokenize text
        let tokens = self.tokenizer.tokenize(text);
        let text_tokens: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
        let (start_mapping, end_mapping) = WhitespaceTokenizer::build_mappings(&tokens);

        // Apply max_len truncation
        let truncated_tokens = if let Some(max_len) = self.max_len {
            if text_tokens.len() > max_len {
                &text_tokens[..max_len]
            } else {
                &text_tokens[..]
            }
        } else {
            &text_tokens[..]
        };

        // Encode schema into token sequences
        let schema_result = self.encode_schema(schema, truncated_tokens.len())?;

        // Build input sequence: [CLS] + schema_tokens + [SEP] + text_tokens + [SEP]
        // Note: Actual tokenization depends on the model's tokenizer
        // For now, we use placeholder token IDs (in real implementation, these would come from the HF tokenizer)
        let mut input_ids: Vec<i64> = Vec::new();
        let mut mapped_indices: Vec<TokenMapping> = Vec::new();
        let mut text_word_indices: Vec<i64> = Vec::new();
        let mut schema_special_indices: Vec<Vec<usize>> = Vec::new();

        // Add schema tokens (placeholder IDs - would be actual token IDs from tokenizer)
        let schema_start = input_ids.len();
        for (schema_idx, schema_tokens) in schema_result.schema_tokens_list.iter().enumerate() {
            let schema_start_pos = input_ids.len();
            for token in schema_tokens {
                // In real implementation, convert token to ID using tokenizer
                // For now, use placeholder
                input_ids.push(self.token_to_id(token));
            }
            let schema_end_pos = input_ids.len();
            schema_special_indices.push((schema_start_pos..schema_end_pos).collect());
        }

        // Add separator
        input_ids.push(self.token_to_id("[SEP]"));

        // Add text tokens
        let text_start = input_ids.len();
        for (token_idx, token) in truncated_tokens.iter().enumerate() {
            input_ids.push(self.token_to_id(token));
            mapped_indices.push(("text".to_string(), 0, token_idx));
            text_word_indices.push(input_ids.len() as i64 - 1);
        }

        // Add final separator
        input_ids.push(self.token_to_id("[SEP]"));

        Ok(ProcessedSample {
            input_ids,
            mapped_indices,
            schema_count: schema_result.schema_tokens_list.len(),
            original_length: truncated_tokens.len() + 2, // +2 for separators
            structure_labels: schema_result.structure_labels,
            task_types: schema_result.task_types,
            text_tokens: truncated_tokens.to_vec(),
            schema_tokens_list: schema_result.schema_tokens_list,
            start_mapping: start_mapping[..truncated_tokens.len()].to_vec(),
            end_mapping: end_mapping[..truncated_tokens.len()].to_vec(),
            original_text: text.to_string(),
            original_schema: schema.clone(),
            text_word_indices,
            text_word_count: truncated_tokens.len(),
            schema_special_indices,
        })
    }

    /// Encode a schema into token sequences.
    fn encode_schema(&self, schema: &JsonValue, text_len: usize) -> Result<SchemaEncodingResult> {
        let mut schema_tokens_list: Vec<Vec<String>> = Vec::new();
        let mut task_types: Vec<String> = Vec::new();
        let mut structure_labels: Vec<JsonValue> = Vec::new();

        // Parse entities
        if let Some(entities) = schema.get("entities") {
            if let Some(entities_obj) = entities.as_object() {
                let entity_names: Vec<String> = entities_obj.keys().cloned().collect();
                if !entity_names.is_empty() {
                    let tokens = self.build_entity_tokens(&entity_names, entities_obj);
                    schema_tokens_list.push(tokens);
                    task_types.push("entities".to_string());
                    structure_labels.push(JsonValue::Array(Vec::new()));
                }
            }
        }

        // Parse classifications
        if let Some(classifications) = schema.get("classifications") {
            if let Some(cls_array) = classifications.as_array() {
                for cls in cls_array {
                    if let Some(cls_obj) = cls.as_object() {
                        if let Some(task) = cls_obj.get("task").and_then(|v| v.as_str()) {
                            let labels = cls_obj
                                .get("labels")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();

                            if !labels.is_empty() {
                                let tokens = self.build_classification_tokens(task, &labels, cls_obj);
                                schema_tokens_list.push(tokens);
                                task_types.push("classifications".to_string());
                                structure_labels.push(JsonValue::Array(Vec::new()));
                            }
                        }
                    }
                }
            }
        }

        // Parse structures
        if let Some(structures) = schema.get("json_structures") {
            if let Some(struct_array) = structures.as_array() {
                for structure in struct_array {
                    if let Some(struct_obj) = structure.as_object() {
                        for (parent, fields) in struct_obj {
                            if let Some(fields_obj) = fields.as_object() {
                                let field_names: Vec<String> = fields_obj.keys().cloned().collect();
                                if !field_names.is_empty() {
                                    let tokens = self.build_structure_tokens(parent, &field_names, fields_obj);
                                    schema_tokens_list.push(tokens);
                                    task_types.push("json_structures".to_string());
                                    structure_labels.push(JsonValue::Array(Vec::new()));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Parse relations
        if let Some(relations) = schema.get("relations") {
            if let Some(rel_array) = relations.as_array() {
                for relation in rel_array {
                    if let Some(rel_obj) = relation.as_object() {
                        for (rel_name, fields) in rel_obj {
                            let field_names: Vec<String> = if let Some(fields_obj) = fields.as_object() {
                                fields_obj.keys().cloned().collect()
                            } else {
                                vec!["head".to_string(), "tail".to_string()]
                            };

                            let tokens = self.build_relation_tokens(rel_name, &field_names);
                            schema_tokens_list.push(tokens);
                            task_types.push("relations".to_string());
                            structure_labels.push(JsonValue::Array(Vec::new()));
                        }
                    }
                }
            }
        }

        Ok(SchemaEncodingResult {
            schema_tokens_list,
            task_types,
            structure_labels,
        })
    }

    /// Build entity schema tokens.
    fn build_entity_tokens(
        &self,
        entity_names: &[String],
        entities_obj: &serde_json::Map<String, JsonValue>,
    ) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            "entities".to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for name in entity_names {
            tokens.push(special_tokens::E_TOKEN.to_string());
            tokens.push(name.clone());

            // Add description if available
            if let Some(desc) = entities_obj.get(name).and_then(|v| v.as_str()) {
                if !desc.is_empty() {
                    tokens.push(special_tokens::DESC_TOKEN.to_string());
                    tokens.push(format!("{name}: {desc}"));
                }
            }
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Build classification schema tokens.
    fn build_classification_tokens(
        &self,
        task: &str,
        labels: &[String],
        cls_obj: &serde_json::Map<String, JsonValue>,
    ) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            task.to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        // Add label descriptions if available
        if let Some(label_descs) = cls_obj.get("label_descriptions").and_then(|v| v.as_object()) {
            for label in labels {
                tokens.push(special_tokens::L_TOKEN.to_string());
                tokens.push(label.clone());

                if let Some(desc) = label_descs.get(label).and_then(|v| v.as_str()) {
                    tokens.push(special_tokens::DESC_TOKEN.to_string());
                    tokens.push(format!("{label}: {desc}"));
                }
            }
        } else {
            for label in labels {
                tokens.push(special_tokens::L_TOKEN.to_string());
                tokens.push(label.clone());
            }
        }

        // Add examples if available
        if let Some(examples) = cls_obj.get("examples").and_then(|v| v.as_array()) {
            for example in examples {
                if let Some(ex_array) = example.as_array() {
                    if ex_array.len() >= 2 {
                        if let (Some(input), Some(output)) =
                            (ex_array[0].as_str(), ex_array[1].as_str())
                        {
                            tokens.push(special_tokens::EXAMPLE_TOKEN.to_string());
                            tokens.push(input.to_string());
                            tokens.push(special_tokens::OUTPUT_TOKEN.to_string());
                            tokens.push(output.to_string());
                        }
                    }
                }
            }
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Build structure schema tokens.
    fn build_structure_tokens(
        &self,
        parent: &str,
        field_names: &[String],
        fields_obj: &serde_json::Map<String, JsonValue>,
    ) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            parent.to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for field in field_names {
            tokens.push(special_tokens::C_TOKEN.to_string());
            tokens.push(field.clone());

            // Add description if available
            if let Some(field_obj) = fields_obj.get(field).and_then(|v| v.as_object()) {
                if let Some(desc) = field_obj.get("description").and_then(|v| v.as_str()) {
                    tokens.push(special_tokens::DESC_TOKEN.to_string());
                    tokens.push(format!("{field}: {desc}"));
                }
            }
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Build relation schema tokens.
    fn build_relation_tokens(&self, rel_name: &str, field_names: &[String]) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            rel_name.to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for field in field_names {
            tokens.push(special_tokens::R_TOKEN.to_string());
            tokens.push(field.clone());
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Convert a token string to token ID.
    ///
    /// In the full implementation, this would use the HuggingFace tokenizer.
    /// For now, this is a placeholder that returns a hash-based ID.
    fn token_to_id(&self, token: &str) -> i64 {
        // Simple hash-based placeholder
        // In real implementation, use self.tokenizer.encode() or similar
        let hash = token.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        (hash % 30522) as i64 // BERT vocab size
    }
}

/// Result of processing a single sample.
#[derive(Debug, Clone)]
struct ProcessedSample {
    input_ids: Vec<i64>,
    mapped_indices: Vec<TokenMapping>,
    schema_count: usize,
    original_length: usize,
    structure_labels: Vec<JsonValue>,
    task_types: Vec<String>,
    text_tokens: Vec<String>,
    schema_tokens_list: Vec<Vec<String>>,
    start_mapping: Vec<usize>,
    end_mapping: Vec<usize>,
    original_text: String,
    original_schema: JsonValue,
    text_word_indices: Vec<i64>,
    text_word_count: usize,
    schema_special_indices: Vec<Vec<usize>>,
}

/// Result of encoding a schema.
#[derive(Debug, Clone)]
struct SchemaEncodingResult {
    schema_tokens_list: Vec<Vec<String>>,
    task_types: Vec<String>,
    structure_labels: Vec<JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> JsonValue {
        serde_json::json!({
            "entities": {
                "person": "Names of people",
                "company": "Organization names"
            },
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative", "neutral"],
                    "cls_threshold": 0.5
                }
            ]
        })
    }

    #[test]
    fn test_collate_single_sample() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = create_test_schema();

        let samples = vec![("Apple CEO Tim Cook", schema)];
        let batch = collator.collate(&samples);

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.batch_size(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_collate_multiple_samples() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = create_test_schema();

        let samples = vec![
            ("Apple CEO Tim Cook", schema.clone()),
            ("Google in Mountain View", schema.clone()),
            ("Microsoft founded by Bill Gates", schema),
        ];

        let batch = collator.collate(&samples);
        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.batch_size(), 3);
    }

    #[test]
    fn test_collate_empty_batch() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let samples: Vec<CollateSample> = vec![];
        let result = collator.collate(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_collate_with_max_len() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::with_max_len(tokenizer, false, Some(5));
        let schema = create_test_schema();

        let long_text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday";
        let samples = vec![(long_text, schema)];
        let batch = collator.collate(&samples);

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        // Text tokens should be truncated to max_len
        assert_eq!(batch.sample_text_tokens(0).unwrap().len(), 5);
    }

    #[test]
    fn test_entity_token_building() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let entities = vec!["person".to_string(), "company".to_string()];
        let entities_obj = serde_json::json!({
            "person": "Names of people",
            "company": "Organization names"
        })
        .as_object()
        .cloned()
        .unwrap();

        let tokens = collator.build_entity_tokens(&entities, &entities_obj);

        assert!(tokens.contains(&"(".to_string()));
        assert!(tokens.contains(&"[P]".to_string()));
        assert!(tokens.contains(&"entities".to_string()));
        assert!(tokens.contains(&"[E]".to_string()));
        assert!(tokens.contains(&"person".to_string()));
        assert!(tokens.contains(&"company".to_string()));
        assert!(tokens.contains(&")".to_string()));
    }

    #[test]
    fn test_classification_token_building() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let labels = vec!["positive".to_string(), "negative".to_string()];
        let cls_obj = serde_json::json!({
            "task": "sentiment",
            "labels": ["positive", "negative"]
        })
        .as_object()
        .cloned()
        .unwrap();

        let tokens = collator.build_classification_tokens("sentiment", &labels, &cls_obj);

        assert!(tokens.contains(&"(".to_string()));
        assert!(tokens.contains(&"[P]".to_string()));
        assert!(tokens.contains(&"sentiment".to_string()));
        assert!(tokens.contains(&"[C]".to_string()));
        assert!(tokens.contains(&"positive".to_string()));
        assert!(tokens.contains(&"negative".to_string()));
    }

    #[test]
    fn test_relation_token_building() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let tokens = collator.build_relation_tokens("works_for", &["head".to_string(), "tail".to_string()]);

        assert!(tokens.contains(&"(".to_string()));
        assert!(tokens.contains(&"[P]".to_string()));
        assert!(tokens.contains(&"works_for".to_string()));
        assert!(tokens.contains(&"[R]".to_string()));
        assert!(tokens.contains(&"head".to_string()));
        assert!(tokens.contains(&"tail".to_string()));
    }

    #[test]
    fn test_schema_encoding_entities_only() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = serde_json::json!({
            "entities": {
                "person": "",
                "company": ""
            }
        });

        let result = collator.encode_schema(&schema, 10);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.task_types, vec!["entities"]);
        assert_eq!(result.schema_tokens_list.len(), 1);
    }

    #[test]
    fn test_schema_encoding_mixed() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = serde_json::json!({
            "entities": {
                "person": ""
            },
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative"]
                }
            ],
            "relations": [
                {
                    "works_for": {
                        "head": "",
                        "tail": ""
                    }
                }
            ]
        });

        let result = collator.encode_schema(&schema, 10);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.task_types.len(), 3);
        assert!(result.task_types.contains(&"entities".to_string()));
        assert!(result.task_types.contains(&"classifications".to_string()));
        assert!(result.task_types.contains(&"relations".to_string()));
    }
}
