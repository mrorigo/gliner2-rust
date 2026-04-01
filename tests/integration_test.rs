//! Integration tests for GLiNER2 Rust.
//!
//! These tests verify the full inference pipeline works end-to-end with
//! real model weights from HuggingFace Hub.

/// Test that the GLiNER2 engine can be created and configured.
///
/// This test verifies the basic setup works without requiring model weights.
#[test]
fn test_gliner2_engine_creation() {
    use gliner2_rust::{GLiNER2, ExtractorConfig};

    // Create config with bert-base-uncased for proper tokenizer compatibility
    let config = ExtractorConfig::new("bert-base-uncased");

    // Create engine - this should work even without model weights
    // (uses random initialization for the encoder)
    let result = GLiNER2::new(&config);
    assert!(result.is_ok(), "Failed to create GLiNER2 engine: {:?}", result.err());

    let engine = result.unwrap();

    // Verify engine properties
    assert_eq!(engine.default_threshold(), 0.5);
    // Device doesn't implement PartialEq, so we just verify it's accessible
    let _device = engine.device();
}

/// Test that schema building works correctly.
#[test]
fn test_schema_building() {
    use gliner2_rust::SchemaBuilder;

    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "company".to_string()])
        .build();

    assert!(schema.is_ok(), "Failed to build schema: {:?}", schema.err());

    let schema = schema.unwrap();
    // Verify schema contains the expected entity types
    let schema_json = serde_json::to_value(&schema).unwrap();
    assert!(schema_json.get("entities").is_some());
}

/// Test entity extraction with a simple schema.
///
/// This test verifies the full pipeline: tokenization → encoding → extraction.
/// Note: Without real model weights, the results will be based on random
/// initialization, but the pipeline should complete without errors.
#[test]
fn test_entity_extraction_pipeline() {
    use gliner2_rust::{GLiNER2, ExtractorConfig};

    // Use bert-base-uncased for proper tokenizer compatibility
    let config = ExtractorConfig::new("bert-base-uncased");
    let engine = GLiNER2::new(&config).expect("Failed to create engine");

    // Run extraction - should complete without errors even with random weights
    let result = engine.extract_entities(
        "Apple CEO Tim Cook",
        &["person", "company"],
        None,
        true,
        true,
        None,
    );
    assert!(result.is_ok(), "Entity extraction failed: {:?}", result.err());
}

/// Test batch entity extraction.
#[test]
fn test_batch_entity_extraction() {
    use gliner2_rust::{GLiNER2, ExtractorConfig};

    // Use bert-base-uncased for proper tokenizer compatibility
    let config = ExtractorConfig::new("bert-base-uncased");
    let engine = GLiNER2::new(&config).expect("Failed to create engine");

    let texts = vec![
        "Apple CEO Tim Cook".to_string(),
        "Google founder Larry Page".to_string(),
    ];

    let result = engine.batch_extract_entities(
        &texts,
        &["person", "company"],
        2,      // batch_size
        None,   // threshold
        1,      // num_workers
        true,   // include_confidence
        true,   // include_spans
        None,   // max_len
    );
    assert!(result.is_ok(), "Batch extraction failed: {:?}", result.err());

    let results = result.unwrap();
    assert_eq!(results.len(), 2);
}

/// Test classification pipeline.
#[test]
fn test_classification_pipeline() {
    use gliner2_rust::{GLiNER2, ExtractorConfig};

    // Use bert-base-uncased for proper tokenizer compatibility
    let config = ExtractorConfig::new("bert-base-uncased");
    let engine = GLiNER2::new(&config).expect("Failed to create engine");

    // classify_text takes tasks as &[(String, Vec<String>)]
    let tasks = vec![
        ("sentiment".to_string(), vec!["positive".to_string(), "negative".to_string()]),
    ];

    let result = engine.classify_text("I love this product!", &tasks, None, false);
    assert!(result.is_ok(), "Classification failed: {:?}", result.err());
}

/// Test that the tokenizer produces valid token IDs.
#[test]
fn test_tokenizer_produces_valid_ids() {
    use gliner2_rust::tokenizer::WhitespaceTokenizer;

    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("Hello world, this is a test.");

    // Should produce multiple tokens
    assert!(!tokens.is_empty());
    assert!(tokens.len() >= 5); // At least: Hello, world, ,, this, is, a, test, .

    // Each token should have valid positions
    for token in &tokens {
        assert!(!token.text.is_empty());
        assert!(token.start < token.end);
    }
}

/// Test that the collator produces valid batches.
#[test]
fn test_collator_produces_valid_batch() {
    use gliner2_rust::batch::ExtractorCollator;
    use gliner2_rust::tokenizer::WhitespaceTokenizer;
    use serde_json::json;

    let tokenizer = WhitespaceTokenizer::new();
    let collator = ExtractorCollator::new(tokenizer, false);

    let schema = json!({
        "entities": {
            "person": "A person's name",
            "company": "A company or organization"
        }
    });

    let samples = vec![
        ("Apple CEO Tim Cook".to_string(), schema.clone()),
        ("Google in Mountain View".to_string(), schema),
    ];

    let batch = collator.collate(&samples);
    assert!(batch.is_ok(), "Collation failed: {:?}", batch.err());

    let batch = batch.unwrap();
    assert_eq!(batch.batch_size(), 2);
    assert!(!batch.input_ids.dims().is_empty());
}

/// Test model configuration validation.
#[test]
fn test_config_validation() {
    use gliner2_rust::ExtractorConfig;

    // Default config should be valid
    let config = ExtractorConfig::default();
    assert!(config.validate().is_ok());

    // Custom config should be valid
    let config = ExtractorConfig::builder()
        .model_name("bert-base-uncased")
        .max_width(12)
        .max_len(Some(384))
        .build();
    assert!(config.is_ok());
}

/// Test that the extractor can run forward pass.
#[test]
fn test_extractor_forward_pass() {
    use gliner2_rust::model::Extractor;
    use gliner2_rust::config::ExtractorConfig;
    use gliner2_rust::batch::PreprocessedBatchBuilder;
    use candle_core::{Device, Tensor};

    let config = ExtractorConfig::default();
    let extractor = Extractor::new(&config).expect("Failed to create extractor");

    // Create a minimal test batch
    let input_ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
    let attention_mask = Tensor::from_slice(&[1u32, 1, 1, 1, 1, 0], (2, 3), &Device::Cpu).unwrap();

    let batch = PreprocessedBatchBuilder::new()
        .input_ids(input_ids)
        .attention_mask(attention_mask)
        .schema_counts(vec![1, 1])
        .task_types(vec![vec!["entities".to_string()], vec!["entities".to_string()]])
        .text_tokens(vec![vec!["apple".to_string(), "inc".to_string()], vec!["hello".to_string()]])
        .schema_tokens_list(vec![
            vec![vec!["(".to_string(), "[P]".to_string(), "entities".to_string()]]
        ])
        .start_mappings(vec![vec![0, 6], vec![0]])
        .end_mappings(vec![vec![5, 10], vec![5]])
        .original_texts(vec!["Apple Inc.".to_string(), "Hello".to_string()])
        .original_schemas(vec![serde_json::Value::Null, serde_json::Value::Null])
        .text_word_counts(vec![2, 1])
        .schema_special_indices(vec![vec![vec![0, 1, 2]], vec![vec![0, 1]]])
        .build()
        .expect("Failed to build batch");

    let output = extractor.forward(&batch);
    assert!(output.is_ok(), "Forward pass failed: {:?}", output.err());

    let output = output.unwrap();
    assert_eq!(output.batch_size, 2);
    assert_eq!(output.token_embeddings.len(), 2);
}
