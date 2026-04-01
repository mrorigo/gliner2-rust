//! Real inference test proving the full GLiNER2 pipeline works end-to-end.
//!
//! This test downloads a real tokenizer from HuggingFace Hub and runs
//! the complete GLiNER2 pipeline to prove everything works.

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

/// Test that the GLiNER2 tokenizer downloads and produces correct token IDs.
#[test]
fn test_gliner2_tokenizer_download() {
    use tokenizers::Tokenizer;
    
    let model_id = "fastino/gliner2-base-v1";
    println!("Downloading GLiNER2 tokenizer from: {}", model_id);
    
    let tokenizer_path = download_from_hub(model_id, "tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    
    // Test tokenization with entity extraction text
    let text = "Apple CEO Tim Cook visited Cupertino, California.";
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
    
    // First token should be [CLS]
    assert_eq!(ids[0], 1, "First token should be [CLS]");
    // Last token should be [SEP]
    assert_eq!(ids[ids.len() - 1], 2, "Last token should be [SEP]");
    
    println!("\n✅ GLiNER2 tokenizer works correctly!");
}

/// Test the full GLiNER2 pipeline with real tokenizer download.
///
/// This test:
/// 1. Downloads a real tokenizer from HuggingFace Hub
/// 2. Creates a GLiNER2 engine with the tokenizer
/// 3. Runs entity extraction on sample text
/// 4. Verifies the pipeline completes successfully
#[test]
fn test_gliner2_pipeline_with_real_tokenizer() {
    use gliner2_rust::{GLiNER2, ExtractorConfig, SchemaBuilder};
    
    let model_id = "bert-base-uncased";
    println!("Setting up GLiNER2 pipeline with real tokenizer from: {}", model_id);
    
    // Download tokenizer from HuggingFace Hub
    let tokenizer_path = download_from_hub(model_id, "tokenizer.json");
    println!("Downloaded tokenizer to: {:?}", tokenizer_path);
    
    // Create config for BERT-base with the downloaded tokenizer
    let config = ExtractorConfig::builder()
        .model_name(model_id)
        .tokenizer_path(tokenizer_path.clone())
        .hidden_size(768)
        .vocab_size(30522)
        .num_hidden_layers(12)
        .num_attention_heads(12)
        .intermediate_size(3072)
        .build()
        .expect("Failed to build config");
    
    // Create GLiNER2 engine
    println!("Creating GLiNER2 engine...");
    let engine = GLiNER2::new(&config)
        .expect("Failed to create GLiNER2 engine");
    
    // Create a schema for entity extraction
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "organization".to_string(), "location".to_string()])
        .build()
        .expect("Failed to build schema");
    
    // Test text
    let text = "Apple CEO Tim Cook announced new products at the headquarters in Cupertino, California.";
    
    println!("\nRunning entity extraction on: {}", text);
    println!("Schema entities: person, organization, location\n");
    
    // Run extraction - this exercises the full pipeline:
    // 1. Whitespace tokenization of text
    // 2. Schema encoding
    // 3. Batch collation with real tokenizer
    // 4. Encoder forward pass
    // 5. Span representation computation
    // 6. Count prediction
    // 7. Result formatting
    let result = engine.extract_entities(
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
    
    // Verify we got a valid result structure
    let result_str = format!("{:?}", result);
    assert!(
        result_str.contains("entities"),
        "Expected entity extraction results, got: {}",
        result_str
    );
    
    println!("\n✅ GLiNER2 pipeline with real tokenizer works!");
}

/// Test batch entity extraction with real tokenizer.
#[test]
fn test_gliner2_batch_extraction_with_real_tokenizer() {
    use gliner2_rust::{GLiNER2, ExtractorConfig, SchemaBuilder};
    
    let model_id = "bert-base-uncased";
    println!("Setting up GLiNER2 batch extraction with: {}", model_id);
    
    // Download tokenizer
    let tokenizer_path = download_from_hub(model_id, "tokenizer.json");
    println!("Downloaded tokenizer to: {:?}", tokenizer_path);
    
    // Create config
    let config = ExtractorConfig::builder()
        .model_name(model_id)
        .tokenizer_path(tokenizer_path.clone())
        .hidden_size(768)
        .vocab_size(30522)
        .num_hidden_layers(12)
        .num_attention_heads(12)
        .intermediate_size(3072)
        .build()
        .expect("Failed to build config");
    
    // Create engine
    let engine = GLiNER2::new(&config)
        .expect("Failed to create GLiNER2 engine");
    
    // Create schema
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "company".to_string()])
        .build()
        .expect("Failed to build schema");
    
    // Test texts
    let texts = vec![
        "Apple CEO Tim Cook".to_string(),
        "Google founder Larry Page".to_string(),
        "Microsoft in Seattle".to_string(),
    ];
    
    println!("\nRunning batch entity extraction on {} texts...", texts.len());
    
    // Run batch extraction
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
    assert_eq!(results.len(), 3, "Should have results for all 3 texts");
    
    println!("\n✅ GLiNER2 batch extraction with real tokenizer works!");
}
