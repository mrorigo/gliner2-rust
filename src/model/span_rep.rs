//! Span representation layer for GLiNER2.
//!
//! This module implements the span representation computation that converts
//! token embeddings into span-level representations. It handles all possible
//! spans up to a maximum width, which is essential for entity, relation,
//! and structure extraction.
//!
//! # Architecture
//!
//! For each token position `i` and span width `w` (0 to max_width-1),
//! the span representation is computed as:
//! - `span_rep[i, w] = f(token_embs[i], token_embs[i + w])`
//!
//! Where `f` is a learned transformation that combines the start and end
//! token embeddings of the span.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::model::SpanRepresentationLayer;
//! use candle_core::{Tensor, Device};
//!
//! let hidden_size = 768;
//! let max_width = 8;
//! let layer = SpanRepresentationLayer::new(hidden_size, max_width, Device::Cpu)?;
//!
//! // Token embeddings: (seq_len, hidden_size)
//! let token_embs = Tensor::randn(0.0f32, 1.0f32, (10, hidden_size), &Device::Cpu)?;
//!
//! let result = layer.forward(&token_embs)?;
//! // span_rep shape: (seq_len, max_width, hidden_size)
//! assert_eq!(result.span_rep.dims()?, &[10, 8, 768]);
//! ```

use candle_core::{Device, DType, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the span representation layer.
#[derive(Clone)]
pub struct SpanRepOutput {
    /// Span representations of shape `(seq_len, max_width, hidden_size)`.
    pub span_rep: Tensor,
    /// Span indices of shape `(seq_len, max_width, 2)` containing [start, end] positions.
    pub spans_idx: Tensor,
    /// Span mask of shape `(seq_len, max_width)` indicating valid spans.
    pub span_mask: Tensor,
}

impl SpanRepOutput {
    /// Create a new span representation output.
    pub fn new(span_rep: Tensor, spans_idx: Tensor, span_mask: Tensor) -> Self {
        Self {
            span_rep,
            spans_idx,
            span_mask,
        }
    }
}

impl std::fmt::Debug for SpanRepOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanRepOutput")
            .field("span_rep_dims", &self.span_rep.dims())
            .field("spans_idx_dims", &self.spans_idx.dims())
            .field("span_mask_dims", &self.span_mask.dims())
            .finish()
    }
}

/// Span representation layer for GLiNER2.
///
/// This layer computes span-level representations from token embeddings
/// by considering all possible spans up to `max_width`. It uses a
/// combination of start token embeddings, end token embeddings, and
/// width embeddings to create rich span representations.
///
/// # Tensor Shapes
///
/// - Input: `(seq_len, hidden_size)`
/// - Output `span_rep`: `(seq_len, max_width, hidden_size)`
/// - Output `spans_idx`: `(seq_len, max_width, 2)`
/// - Output `span_mask`: `(seq_len, max_width)`
pub struct SpanRepresentationLayer {
    /// Maximum span width.
    pub max_width: usize,
    /// Hidden size of the model.
    pub hidden_size: usize,
    /// Width embedding layer.
    width_embedding: Embedding,
    /// Layer normalization for span representations.
    layer_norm: Option<candle_nn::LayerNorm>,
    /// Linear projection for span representations (optional).
    span_linear: Option<Linear>,
    /// Device for tensor operations.
    device: Device,
}

impl std::fmt::Debug for SpanRepresentationLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanRepresentationLayer")
            .field("max_width", &self.max_width)
            .field("hidden_size", &self.hidden_size)
            .field("device", &self.device)
            .finish()
    }
}

impl Clone for SpanRepresentationLayer {
    fn clone(&self) -> Self {
        Self {
            max_width: self.max_width,
            hidden_size: self.hidden_size,
            width_embedding: self.width_embedding.clone(),
            layer_norm: self.layer_norm.clone(),
            span_linear: self.span_linear.clone(),
            device: self.device.clone(),
        }
    }
}

impl SpanRepresentationLayer {
    /// Create a new span representation layer with random initialization.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The hidden size of the model.
    /// * `max_width` - The maximum span width to consider.
    /// * `device` - The device to place tensors on.
    ///
    /// # Returns
    ///
    /// A new `SpanRepresentationLayer`.
    pub fn new(hidden_size: usize, max_width: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create width embedding layer
        let width_embedding = candle_nn::embedding(max_width, hidden_size, vb.pp("width_embedding"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create width embedding: {e}")))?;

        Ok(Self {
            max_width,
            hidden_size,
            width_embedding,
            layer_norm: None,
            span_linear: None,
            device,
        })
    }

    /// Create a span representation layer from a config.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `device` - The device to place tensors on.
    pub fn from_config(config: &ExtractorConfig, device: Device) -> Result<Self> {
        Self::new(config.hidden_size, config.max_width, device)
    }

    /// Create a span representation layer loaded from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `vb` - The VarBuilder containing the weights.
    /// * `config` - The extractor configuration.
    /// * `device` - The device to place tensors on.
    pub fn from_var_builder(
        vb: VarBuilder,
        config: &ExtractorConfig,
        device: Device,
    ) -> Result<Self> {
        let span_path = vb.pp("span_rep");

        // Load width embedding
        let width_embedding = candle_nn::embedding(config.max_width, config.hidden_size, span_path.pp("width_embedding"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to load width embedding: {e}")))?;

        // Load layer norm if present
        let layer_norm = if span_path.contains_tensor("layer_norm.weight") {
            let ln = candle_nn::layer_norm(config.hidden_size, 1e-5, span_path.pp("layer_norm"))
                .map_err(|e| GlinerError::model_loading(format!("Failed to load layer norm: {e}")))?;
            Some(ln)
        } else {
            None
        };

        // Load span linear if present
        let span_linear = if span_path.contains_tensor("span_linear.weight") {
            let linear = candle_nn::linear(config.hidden_size * 2, config.hidden_size, span_path.pp("span_linear"))
                .map_err(|e| GlinerError::model_loading(format!("Failed to load span linear: {e}")))?;
            Some(linear)
        } else {
            None
        };

        Ok(Self {
            max_width: config.max_width,
            hidden_size: config.hidden_size,
            width_embedding,
            layer_norm,
            span_linear,
            device,
        })
    }

    /// Forward pass: compute span representations from token embeddings.
    ///
    /// # Arguments
    ///
    /// * `token_embs` - Token embeddings of shape `(seq_len, hidden_size)`.
    ///
    /// # Returns
    ///
    /// A `SpanRepOutput` containing span representations, indices, and mask.
    pub fn forward(&self, token_embs: &Tensor) -> Result<SpanRepOutput> {
        let dims = token_embs.dims();

        if dims.len() != 2 {
            return Err(GlinerError::dimension_mismatch(
                vec![2],
                dims.to_vec(),
            ));
        }

        let seq_len = dims[0];
        let hidden_size = dims[1];

        if hidden_size != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(
                vec![self.hidden_size],
                vec![hidden_size],
            ));
        }

        // Compute span representations for each width
        let mut span_reps: Vec<Tensor> = Vec::with_capacity(self.max_width);

        for width in 0..self.max_width {
            let end_idx = seq_len.saturating_sub(width);

            if end_idx == 0 {
                // No valid spans for this width - create zeros padded to seq_len
                let zeros = Tensor::zeros((seq_len, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to create zeros tensor: {e}")))?;
                span_reps.push(zeros);
                continue;
            }

            // Get start tokens: token_embs[0:end_idx]
            let start_tokens = token_embs.narrow(0, 0, end_idx)
                .map_err(|e| GlinerError::model_loading(format!("Failed to narrow start tokens: {e}")))?;

            // Get end tokens: token_embs[width:width+end_idx]
            let end_tokens = token_embs.narrow(0, width, end_idx)
                .map_err(|e| GlinerError::model_loading(format!("Failed to narrow end tokens: {e}")))?;

            // Get width embedding
            let width_idx = Tensor::new(&[width as u32], &self.device)
                .map_err(|e| GlinerError::model_loading(format!("Failed to create width index tensor: {e}")))?;
            let width_emb = self.width_embedding.forward(&width_idx)
                .map_err(|e| GlinerError::model_loading(format!("Width embedding forward failed: {e}")))?;

            // Combine start + end + width embedding
            let span_rep = if let Some(linear) = &self.span_linear {
                // With linear projection: cat(start, end) -> linear
                let combined = Tensor::cat(&[&start_tokens, &end_tokens], 1)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to concat tokens: {e}")))?;
                linear.forward(&combined)
                    .map_err(|e| GlinerError::model_loading(format!("Span linear forward failed: {e}")))?
            } else {
                // Simple addition: start + end + width_emb
                let combined = (&start_tokens + &end_tokens)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to add tokens: {e}")))?;
                // Broadcast width_emb to match shape
                let width_emb_broadcasted = width_emb.broadcast_as((end_idx, self.hidden_size))
                    .map_err(|e| GlinerError::model_loading(format!("Failed to broadcast width embedding: {e}")))?;
                (&combined + &width_emb_broadcasted)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to add width embedding: {e}")))?
            };

            // Pad to seq_len if needed
            let span_rep_dims = span_rep.dims();
            if span_rep_dims[0] < seq_len {
                let pad_size = seq_len - span_rep_dims[0];
                let padding = Tensor::zeros((pad_size, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to create padding: {e}")))?;
                let padded = Tensor::cat(&[&span_rep, &padding], 0)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to pad span rep: {e}")))?;
                span_reps.push(padded);
            } else {
                span_reps.push(span_rep);
            }
        }

        // Stack all width representations: (seq_len, max_width, hidden_size)
        let span_rep = Tensor::stack(&span_reps, 1)
            .map_err(|e| GlinerError::model_loading(format!("Failed to stack span reps: {e}")))?;

        // Apply layer norm if present
        let span_rep = if let Some(ln) = &self.layer_norm {
            // Layer norm over the last dimension
            let normalized = span_rep
                .to_dtype(DType::F32)
                .map_err(|e| GlinerError::model_loading(format!("Failed to convert to F32: {e}")))?;
            Module::forward(ln, &normalized)
                .map_err(|e| GlinerError::model_loading(format!("Layer norm forward failed: {e}")))?
        } else {
            span_rep
        };

        // Create span indices: (seq_len, max_width, 2)
        let mut spans_idx_data: Vec<u32> = Vec::with_capacity(seq_len * self.max_width * 2);
        let mut span_mask_data: Vec<u32> = Vec::with_capacity(seq_len * self.max_width);

        for i in 0..seq_len {
            for w in 0..self.max_width {
                let end = i + w;
                let valid = if end < seq_len { 1 } else { 0 };
                spans_idx_data.push(i as u32);
                spans_idx_data.push(end as u32);
                span_mask_data.push(valid);
            }
        }

        let spans_idx = Tensor::from_slice(&spans_idx_data, (seq_len, self.max_width, 2), &self.device)
            .map_err(|e| GlinerError::model_loading(format!("Failed to create spans_idx: {e}")))?;

        let span_mask = Tensor::from_slice(&span_mask_data, (seq_len, self.max_width), &self.device)
            .map_err(|e| GlinerError::model_loading(format!("Failed to create span_mask: {e}")))?;

        Ok(SpanRepOutput::new(span_rep, spans_idx, span_mask))
    }

    /// Forward pass for a batch of token embeddings.
    ///
    /// # Arguments
    ///
    /// * `token_embs_list` - List of token embeddings, each of shape `(seq_len, hidden_size)`.
    ///
    /// # Returns
    ///
    /// A vector of `SpanRepOutput`, one for each input.
    pub fn forward_batch(&self, token_embs_list: &[Tensor]) -> Result<Vec<SpanRepOutput>> {
        token_embs_list
            .iter()
            .map(|embs| self.forward(embs))
            .collect()
    }

    /// Compute span representations with attention scoring.
    ///
    /// This method is used during inference to compute span scores
    /// for entity/relation/structure extraction.
    ///
    /// # Arguments
    ///
    /// * `token_embs` - Token embeddings of shape `(seq_len, hidden_size)`.
    /// * `schema_embs` - Schema embeddings of shape `(num_schemas, hidden_size)`.
    ///
    /// # Returns
    ///
    /// Span scores of shape `(num_schemas, seq_len, max_width)`.
    pub fn compute_span_scores(
        &self,
        token_embs: &Tensor,
        schema_embs: &Tensor,
    ) -> Result<Tensor> {
        let span_output = self.forward(token_embs)?;
        let span_rep = &span_output.span_rep;
        let span_mask = &span_output.span_mask;

        let dims = span_rep.dims();

        if dims.len() != 3 {
            return Err(GlinerError::dimension_mismatch(
                vec![3],
                dims.to_vec(),
            ));
        }

        let seq_len = dims[0];
        let max_width = dims[1];
        let num_schemas = schema_embs.dims()[0];

        // Reshape span_rep to (seq_len * max_width, hidden_size)
        let span_flat = span_rep.reshape((seq_len * max_width, self.hidden_size))
            .map_err(|e| GlinerError::model_loading(format!("Failed to reshape span_rep: {e}")))?;

        // Compute scores: schema_embs @ span_flat.T
        // Result: (num_schemas, seq_len * max_width)
        let span_flat_t = span_flat.transpose(0, 1)
            .map_err(|e| GlinerError::model_loading(format!("Failed to transpose span_flat: {e}")))?;
        let scores = schema_embs.matmul(&span_flat_t)
            .map_err(|e| GlinerError::model_loading(format!("Failed to compute span scores: {e}")))?;

        // Reshape to (num_schemas, seq_len, max_width)
        let scores = scores.reshape((num_schemas, seq_len, max_width))
            .map_err(|e| GlinerError::model_loading(format!("Failed to reshape scores: {e}")))?;

        // Apply mask (set invalid spans to very negative value)
        let mask_unsqueezed = span_mask.unsqueeze(0)
            .map_err(|e| GlinerError::model_loading(format!("Failed to unsqueeze mask: {e}")))?;
        let mask_expanded = mask_unsqueezed
            .broadcast_as((num_schemas, seq_len, max_width))
            .map_err(|e| GlinerError::model_loading(format!("Failed to expand mask: {e}")))?;

        let neg_inf = Tensor::full(-1e9f32, (num_schemas, seq_len, max_width), &self.device)
            .map_err(|e| GlinerError::model_loading(format!("Failed to create neg_inf: {e}")))?;

        let masked_scores = mask_expanded.where_cond(&scores, &neg_inf)
            .map_err(|e| GlinerError::model_loading(format!("Failed to apply mask: {e}")))?;

        Ok(masked_scores)
    }

    /// Get the device the layer is on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Builder for constructing `SpanRepresentationLayer` with custom settings.
#[derive(Debug, Clone)]
pub struct SpanRepBuilder {
    hidden_size: usize,
    max_width: usize,
    device: Device,
    use_layer_norm: bool,
    use_linear_projection: bool,
}

impl Default for SpanRepBuilder {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            max_width: 8,
            device: Device::Cpu,
            use_layer_norm: true,
            use_linear_projection: false,
        }
    }
}

impl SpanRepBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hidden size.
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Set the maximum span width.
    pub fn max_width(mut self, width: usize) -> Self {
        self.max_width = width;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Enable layer normalization.
    pub fn use_layer_norm(mut self, enabled: bool) -> Self {
        self.use_layer_norm = enabled;
        self
    }

    /// Enable linear projection.
    pub fn use_linear_projection(mut self, enabled: bool) -> Self {
        self.use_linear_projection = enabled;
        self
    }

    /// Build the span representation layer.
    pub fn build(self) -> Result<SpanRepresentationLayer> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);

        let width_embedding = candle_nn::embedding(self.max_width, self.hidden_size, vb.pp("width_embedding"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create width embedding: {e}")))?;

        let layer_norm = if self.use_layer_norm {
            let ln = candle_nn::layer_norm(self.hidden_size, 1e-5, vb.pp("layer_norm"))
                .map_err(|e| GlinerError::model_loading(format!("Failed to create layer norm: {e}")))?;
            Some(ln)
        } else {
            None
        };

        let span_linear = if self.use_linear_projection {
            let linear = candle_nn::linear(self.hidden_size * 2, self.hidden_size, vb.pp("span_linear"))
                .map_err(|e| GlinerError::model_loading(format!("Failed to create span linear: {e}")))?;
            Some(linear)
        } else {
            None
        };

        Ok(SpanRepresentationLayer {
            max_width: self.max_width,
            hidden_size: self.hidden_size,
            width_embedding,
            layer_norm,
            span_linear,
            device: self.device,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_rep_layer_creation() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 768);
        assert_eq!(layer.max_width, 8);
    }

    #[test]
    fn test_span_rep_forward() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(0.0f32, 1.0f32, (10, 768), &Device::Cpu).unwrap();

        let result = layer.forward(&token_embs);
        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.span_rep.dims(), &[10, 8, 768]);
        assert_eq!(result.spans_idx.dims(), &[10, 8, 2]);
        assert_eq!(result.span_mask.dims(), &[10, 8]);
    }

    #[test]
    fn test_span_rep_forward_single_token() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(0.0f32, 1.0f32, (1, 768), &Device::Cpu).unwrap();

        let result = layer.forward(&token_embs).unwrap();
        assert_eq!(result.span_rep.dims(), &[1, 8, 768]);
        assert_eq!(result.spans_idx.dims(), &[1, 8, 2]);
        assert_eq!(result.span_mask.dims(), &[1, 8]);

        // Only width 0 should be valid for single token
        let mask_data: Vec<u32> = result.span_mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask_data[0], 1); // width 0 is valid
        assert_eq!(mask_data[1], 0); // width 1 is invalid
    }

    #[test]
    fn test_span_rep_forward_empty() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::zeros((0, 768), DType::F32, &Device::Cpu).unwrap();

        // Empty input should succeed with empty output
        let result = layer.forward(&token_embs);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.span_rep.dims(), &[0, 8, 768]);
        assert_eq!(result.spans_idx.dims(), &[0, 8, 2]);
        assert_eq!(result.span_mask.dims(), &[0, 8]);
    }

    #[test]
    fn test_span_rep_forward_wrong_hidden_size() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(0.0f32, 1.0f32, (10, 512), &Device::Cpu).unwrap();

        let result = layer.forward(&token_embs);
        assert!(result.is_err());
    }

    #[test]
    fn test_span_rep_forward_batch() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let embs_list = vec![
            Tensor::randn(0.0f32, 1.0f32, (10, 768), &Device::Cpu).unwrap(),
            Tensor::randn(0.0f32, 1.0f32, (5, 768), &Device::Cpu).unwrap(),
        ];

        let results = layer.forward_batch(&embs_list);
        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].span_rep.dims(), &[10, 8, 768]);
        assert_eq!(results[1].span_rep.dims(), &[5, 8, 768]);
    }

    #[test]
    fn test_span_rep_compute_span_scores() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(0.0f32, 1.0f32, (10, 768), &Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap();

        let scores = layer.compute_span_scores(&token_embs, &schema_embs);
        assert!(scores.is_ok());
        let scores = scores.unwrap();
        assert_eq!(scores.dims(), &[3, 10, 8]);
    }

    #[test]
    fn test_span_rep_builder() {
        let layer = SpanRepBuilder::new()
            .hidden_size(512)
            .max_width(12)
            .device(Device::Cpu)
            .use_layer_norm(true)
            .use_linear_projection(false)
            .build();

        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 512);
        assert_eq!(layer.max_width, 12);
    }

    #[test]
    fn test_span_rep_deterministic() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(0.0f32, 1.0f32, (10, 768), &Device::Cpu).unwrap();

        let result1 = layer.forward(&token_embs).unwrap();
        let result2 = layer.forward(&token_embs).unwrap();

        // Results should be identical for same input
        let diff = (&result1.span_rep - &result2.span_rep).unwrap();
        let max_diff = diff.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff < 1e-6);
    }
}
