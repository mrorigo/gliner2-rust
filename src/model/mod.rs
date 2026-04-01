//! Model module for GLiNER2.
//!
//! This module provides the neural network components for the GLiNER2 model,
//! including the main `Extractor` model, span representation layer, count
//! prediction layers, classification head, and model weight loading utilities.
//!
//! # Architecture
//!
//! The GLiNER2 model consists of:
//! - **Encoder**: Transformer encoder (BERT-like) for token embeddings
//! - **Span Representation**: Computes span representations from token embeddings
//! - **Count Prediction**: Predicts number of instances for each schema
//! - **Classifier**: Classification head for classification tasks
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::model::{Extractor, ExtractorConfig};
//!
//! let config = ExtractorConfig::default();
//! let model = Extractor::new(&config)?;
//! model.load_weights("path/to/weights.safetensors")?;
//! ```

pub mod classifier;
pub mod count_pred;
pub mod extractor;
pub mod loading;
pub mod span_rep;

// Re-export main types
pub use classifier::ClassifierHead;
pub use count_pred::CountPredictionLayer;
pub use extractor::{Extractor, ExtractorOutput};
pub use loading::ModelLoader;
pub use span_rep::SpanRepresentationLayer;
