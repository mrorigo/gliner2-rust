//! Weight name mapping for GLiNER2 model loading.
//!
//! This module provides utilities for mapping GLiNER2 weight names from
//! HuggingFace safetensors format to candle's expected format. It handles
//! the differences between GLiNER2's custom architecture and standard
//! BERT/DeBERTa implementations.
//!
//! # Weight Name Mapping
//!
//! GLiNER2 uses custom weight names that need to be mapped to candle's
//! expected format for each component.

use std::collections::HashMap;

/// Maps GLiNER2 weight names to candle component names.
///
/// Returns a mapping from GLiNER2 weight names to candle VarBuilder paths.
pub fn build_weight_map() -> HashMap<String, String> {
    let mut map = HashMap::new();

    // Encoder weights (DeBERTa v3)
    // GLiNER2 stores: encoder.embeddings.word_embeddings.weight
    // Candle expects: embeddings.word_embeddings.weight (after "encoder" prefix is added)
    // The "encoder" prefix is handled by VarBuilder.pp("encoder")

    // Span representation layer
    // GLiNER2: span_rep.span_rep_layer.project_start.0.weight
    // Our impl: span_rep.project_start (Linear layer)
    map.insert(
        "span_rep.span_rep_layer.project_start.0.weight".to_string(),
        "span_rep.project_start.weight".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_start.0.bias".to_string(),
        "span_rep.project_start.bias".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_start.3.weight".to_string(),
        "span_rep.project_start_ln.weight".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_start.3.bias".to_string(),
        "span_rep.project_start_ln.bias".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_end.0.weight".to_string(),
        "span_rep.project_end.weight".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_end.0.bias".to_string(),
        "span_rep.project_end.bias".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_end.3.weight".to_string(),
        "span_rep.project_end_ln.weight".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.project_end.3.bias".to_string(),
        "span_rep.project_end_ln.bias".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.out_project.0.weight".to_string(),
        "span_rep.out_project.0.weight".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.out_project.0.bias".to_string(),
        "span_rep.out_project.0.bias".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.out_project.3.weight".to_string(),
        "span_rep.out_project.1.weight".to_string(),
    );
    map.insert(
        "span_rep.span_rep_layer.out_project.3.bias".to_string(),
        "span_rep.out_project.1.bias".to_string(),
    );

    // Classifier head
    // GLiNER2: classifier.0.weight, classifier.0.bias, classifier.2.weight, classifier.2.bias
    // Our impl: classifier (Linear layer: hidden_size -> 1)
    // The GLiNER2 classifier is a 3-layer MLP, we use a single Linear layer
    // For now, map the final layer
    map.insert(
        "classifier.2.weight".to_string(),
        "classifier.weight".to_string(),
    );
    map.insert(
        "classifier.2.bias".to_string(),
        "classifier.bias".to_string(),
    );

    // Count prediction
    // GLiNER2: count_pred.0.weight, count_pred.0.bias, count_pred.2.weight, count_pred.2.bias
    // Our impl: count_pred (Linear layer: hidden_size -> max_count)
    map.insert(
        "count_pred.2.weight".to_string(),
        "count_pred.linear.weight".to_string(),
    );
    map.insert(
        "count_pred.2.bias".to_string(),
        "count_pred.linear.bias".to_string(),
    );

    // Count embedding (GRU-based)
    // GLiNER2: count_embed.gru.weight_ih_l0, count_embed.gru.weight_hh_l0, etc.
    // Our impl: count_embedding (Embedding layer)
    // This is a significant architectural difference - GLiNER2 uses GRU, we use Embedding
    // For now, skip count embedding weights as they require architectural changes

    map
}

/// Check if a weight name belongs to the encoder.
pub fn is_encoder_weight(name: &str) -> bool {
    name.starts_with("encoder.")
}

/// Check if a weight name belongs to the span representation layer.
pub fn is_span_rep_weight(name: &str) -> bool {
    name.starts_with("span_rep.")
}

/// Check if a weight name belongs to the classifier.
pub fn is_classifier_weight(name: &str) -> bool {
    name.starts_with("classifier.")
}

/// Check if a weight name belongs to count prediction.
pub fn is_count_pred_weight(name: &str) -> bool {
    name.starts_with("count_pred.") || name.starts_with("count_embed.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_map_creation() {
        let map = build_weight_map();
        assert!(!map.is_empty());
        assert!(map.contains_key("span_rep.span_rep_layer.project_start.0.weight"));
        assert!(map.contains_key("classifier.2.weight"));
        assert!(map.contains_key("count_pred.2.weight"));
    }

    #[test]
    fn test_weight_classification() {
        assert!(is_encoder_weight(
            "encoder.embeddings.word_embeddings.weight"
        ));
        assert!(is_span_rep_weight(
            "span_rep.span_rep_layer.project_start.0.weight"
        ));
        assert!(is_classifier_weight("classifier.2.weight"));
        assert!(is_count_pred_weight("count_pred.2.weight"));
        assert!(is_count_pred_weight("count_embed.gru.weight_ih_l0"));
    }
}
