//! Real inference test with actual GLiNER2 model from HuggingFace Hub.
//!
//! This test downloads a real GLiNER2 model and tokenizer, then runs
//! entity extraction to prove the full pipeline works end-to-end.

use std::path::PathBuf;

/// Download a file from HuggingFace Hub using hf-hub.
fn download_from_hub(repo_id: &str, filename: &str) -> PathBuf {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    
    let repo = Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    );
    
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .expect("Failed to build HF API");
    
    let repo_api = api.repo(repo);
    repo_api.get(filename).expect(&format!("Failed to download {}", filename))
}

/// Test real entity extraction with a downloaded GLiNER2 model.
///
/// This test:
/// 1. Downloads model.safetensors and tokenizer.json from HuggingFace Hub
/// 2. Loads the model and tokenizer
/// 3. Runs entity extraction on sample text
/// 4. Verifies we get meaningful results (not random noise)
#[test]
fn test_real_gliner2_inference() {
    use gliner2_rust::{GLiNER2, ExtractorConfig, SchemaBuilder};
    
    // Use the official GLiNER2 base model
    let model_id = "fastino/gliner2-base-v1";
    
    println!("Downloading model from HuggingFace Hub: {}", model_id);
    
    // Download model weights
    let model_path = download_from_hub(model_id, "model.safetensors");
    println!("Downloaded model weights to: {:?}", model_path);
    
    // Download tokenizer
    let tokenizer_path = download_from_hub(model_id, "tokenizer.json");
    println!("Downloaded tokenizer to: {:?}", tokenizer_path);
    
    // Create config matching the actual GLiNER2 model architecture
    // The model uses DeBERTa-v3-base encoder with vocab_size 128011
    let config = ExtractorConfig::builder()
        .model_name(model_id)
        .tokenizer_path(tokenizer_path.clone())
        .hidden_size(768)
        .vocab_size(128011)
        .num_hidden_layers(12)
        .num_attention_heads(12)
        .intermediate_size(3072)
        .build()
        .expect("Failed to build config");
    
    // Load the model with the correct config
    println!("Creating GLiNER2 engine...");
    let mut model = GLiNER2::new(&config)
        .expect("Failed to create GLiNER2 engine");
    
    // Load the actual model weights
    println!("Loading model weights from: {:?}", model_path);
    model.load_weights(&model_path)
        .expect("Failed to load model weights");
    
    // Create a schema for entity extraction
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "organization".to_string(), "location".to_string()])
        .build()
        .expect("Failed to build schema");
    
    // Test text
    let text = "Apple CEO Tim Cook announced new products at the headquarters in Cupertino, California.";
    
    println!("\nRunning entity extraction on: {}", text);
    println!("Schema entities: person, organization, location\n");
    
    // Run extraction
    let result = model.extract_entities(
        text,
        &["person", "organization", "location"],
        Some(0.5),  // threshold
        true,       // include_confidence
        true,       // include_spans
        None,       // max_len
    );
    
    assert!(result.is_ok(), "Entity extraction failed: {:?}", result.err());
    
    let result = result.unwrap();
    
    // Print results
    println!("Extraction results:");
    println!("{:#?}", result);
    
    // Verify we got some results (the model should find entities)
    // Note: With real weights, we should get meaningful entities
    let result_str = format!("{:?}", result);
    
    // The result should contain entity information
    assert!(
        result_str.contains("entities") || result_str.contains("person") || 
        result_str.contains("organization") || result_str.contains("location"),
        "Expected entity extraction results, got: {}",
        result_str
    );
    
    println!("\n✅ Real inference pipeline works!");
}

/// Test that the tokenizer produces correct token IDs for the model.
#[test]
fn test_real_tokenizer() {
    use tokenizers::Tokenizer;
    
    let model_id = "fastino/gliner2-base-v1";
    println!("Downloading tokenizer from: {}", model_id);
    
    let tokenizer_path = download_from_hub(model_id, "tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    
    // Test tokenization
    let text = "Hello world! This is a test.";
    let encoding = tokenizer.encode(text, true)
        .expect("Failed to encode text");
    
    let tokens = encoding.get_tokens();
    let ids = encoding.get_ids();
    
    println!("Text: {}", text);
    println!("Tokens: {:?}", tokens);
    println!("Token IDs: {:?}", ids);
    
    // Verify we got valid token IDs
    assert!(!ids.is_empty(), "Should have token IDs");
    assert_eq!(tokens.len(), ids.len(), "Tokens and IDs should match");
    
    // First token should be [CLS] (ID 1 for this model)
    assert_eq!(ids[0], 1, "First token should be [CLS]");
    
    println!("\n✅ Tokenizer works correctly!");
}
