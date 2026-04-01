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
//! use tch::{Tensor, Device, Kind};
//!
//! let hidden_size = 768;
//! let max_width = 8;
//! let layer = SpanRepresentationLayer::new(hidden_size, max_width, Device::Cpu)?;
//!
//! // Token embeddings: (seq_len, hidden_size)
//! let token_embs = Tensor::randn([10, hidden_size], (Kind::Float, Device::Cpu));
//!
//! let result = layer.forward(&token_embs)?;
//! // span_rep shape: (seq_len, max_width, hidden_size)
//! assert_eq!(result.span_rep.size(), &[10, 8, 768]);
//! ```

use tch::{nn, nn::Module, Device, Kind, Tensor};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the span representation layer.
#[derive(Debug)]
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
#[derive(Debug)]
pub struct SpanRepresentationLayer {
    /// Maximum span width.
    pub max_width: usize,
    /// Hidden size of the model.
    pub hidden_size: usize,
    /// Width embedding layer.
    width_embedding: Tensor,
    /// Layer normalization for span representations.
    layer_norm: Option<Tensor>,
    /// Linear projection for span representations.
    span_linear_w: Option<Tensor>,
    span_linear_b: Option<Tensor>,
    /// Device for tensor operations.
    device: Device,
}

impl SpanRepresentationLayer {
    /// Create a new span representation layer.
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
        // Initialize width embeddings
        let width_embedding = Tensor::randn(
            &[max_width as i64, hidden_size as i64],
            (Kind::Float, device),
        ) * 0.02;

        Ok(Self {
            max_width,
            hidden_size,
            width_embedding,
            layer_norm: None,
            span_linear_w: None,
            span_linear_b: None,
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

    /// Initialize the layer with weights from a VarStore.
    ///
    /// # Arguments
    ///
    /// * `vs` - The variable store containing the weights.
    /// * `prefix` - The prefix for weight names.
    pub fn init_from_varstore(&mut self, vs: &nn::Path, prefix: &str) -> Result<()> {
        let span_path = vs / prefix;

        // Load width embeddings
        if let Ok(weights) = span_path.var("width_embedding", &[self.max_width as i64, self.hidden_size as i64], Default::default()) {
            self.width_embedding = weights;
        }

        // Load layer norm if present
        if let Ok(weight) = span_path.var("layer_norm.weight", &[self.hidden_size as i64], Default::default()) {
            if let Ok(bias) = span_path.var("layer_norm.bias", &[self.hidden_size as i64], Default::default()) {
                self.layer_norm = Some(Tensor::cat(&[&weight, &bias], 0));
            }
        }

        // Load span linear if present
        if let Ok(w) = span_path.var("span_linear.weight", &[self.hidden_size as i64, self.hidden_size as i64], Default::default()) {
            if let Ok(b) = span_path.var("span_linear.bias", &[self.hidden_size as i64], Default::default()) {
                self.span_linear_w = Some(w);
                self.span_linear_b = Some(b);
            }
        }

        Ok(())
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
        let size = token_embs.size();
        if size.len() != 2 {
            return Err(GlinerError::dimension_mismatch(
                vec![2],
                size.iter().map(|d| *d as usize).collect(),
            ));
        }

        let seq_len = size[0] as usize;
        let hidden_size = size[1] as usize;

        if hidden_size != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(
                vec![self.hidden_size],
                vec![hidden_size],
            ));
        }

        // Compute span representations
        let mut span_reps: Vec<Tensor> = Vec::with_capacity(self.max_width);

        for width in 0..self.max_width {
            // Get start tokens: token_embs[0:seq_len-width]
            let end_idx = seq_len - width;
            if end_idx == 0 {
                // No valid spans for this width
                let zeros = Tensor::zeros(&[0, self.hidden_size as i64], (Kind::Float, self.device));
                span_reps.push(zeros);
                continue;
            }

            let start_tokens = token_embs.narrow(0, 0, end_idx as i64);
            // Get end tokens: token_embs[width:seq_len]
            let end_tokens = token_embs.narrow(0, width as i64, end_idx as i64);

            // Get width embedding
            let width_emb = self.width_embedding.narrow(0, width as i64, 1);

            // Combine: start + end + width_embedding
            // In the Python implementation, this is typically:
            // span_rep = start_tokens + end_tokens + width_emb
            // Or with learned projection
            let span_rep = if let (Some(w), Some(b)) = (&self.span_linear_w, &self.span_linear_b) {
                // With linear projection
                let combined = Tensor::cat(&[&start_tokens, &end_tokens], 1);
                combined.matmul(&w.t()).add(b)
            } else {
                // Simple addition
                let combined = &start_tokens + &end_tokens;
                combined + width_emb
            };

            // Pad to seq_len if needed
            if span_rep.size()[0] as usize < seq_len {
                let pad_size = seq_len - span_rep.size()[0] as usize;
                let padding = Tensor::zeros(&[pad_size as i64, self.hidden_size as i64], (Kind::Float, self.device));
                let padded = Tensor::cat(&[&span_rep, &padding], 0);
                span_reps.push(padded);
            } else {
                span_reps.push(span_rep);
            }
        }

        // Stack all width representations: (seq_len, max_width, hidden_size)
        let span_rep = Tensor::stack(&span_reps, 1);

        // Apply layer norm if present
        let span_rep = if let Some(ln) = &self.layer_norm {
            let weight = ln.narrow(0, 0, self.hidden_size as i64);
            let bias = ln.narrow(0, self.hidden_size as i64, self.hidden_size as i64);
            span_rep.layer_norm(&[self.hidden_size as i64], Some(&weight), Some(&bias), 1e-5)
        } else {
            span_rep
        };

        // Create span indices: (seq_len, max_width, 2)
        let mut spans_idx_data: Vec<i64> = Vec::with_capacity(seq_len * self.max_width * 2);
        let mut span_mask_data: Vec<i64> = Vec::with_capacity(seq_len * self.max_width);

        for i in 0..seq_len {
            for w in 0..self.max_width {
                let end = i + w;
                let valid = if end < seq_len { 1 } else { 0 };
                spans_idx_data.push(i as i64);
                spans_idx_data.push(end as i64);
                span_mask_data.push(valid);
            }
        }

        let spans_idx = Tensor::from_slice(&spans_idx_data)
            .view(&[seq_len as i64, self.max_width as i64, 2])
            .to_device(self.device)
            .to_kind(Kind::Int64);

        let span_mask = Tensor::from_slice(&span_mask_data)
            .view(&[seq_len as i64, self.max_width as i64])
            .to_device(self.device)
            .to_kind(Kind::Int64);

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

        // Compute attention scores: einsum("lkd,bpd->bplk", span_rep, schema_embs)
        // In Rust/tch, we need to reshape and use matmul
        // span_rep: (seq_len, max_width, hidden_size)
        // schema_embs: (num_schemas, hidden_size)
        // Output: (num_schemas, seq_len, max_width)

        let seq_len = span_rep.size()[0];
        let max_width = span_rep.size()[1];
        let num_schemas = schema_embs.size()[0];

        // Reshape span_rep to (seq_len * max_width, hidden_size)
        let span_flat = span_rep.view(&[seq_len * max_width, self.hidden_size as i64]);

        // Compute scores: schema_embs @ span_flat.T
        // Result: (num_schemas, seq_len * max_width)
        let scores = schema_embs.matmul(&span_flat.t());

        // Reshape to (num_schemas, seq_len, max_width)
        let scores = scores.view(&[num_schemas, seq_len, max_width]);

        // Apply mask (set invalid spans to very negative value)
        let mask_expanded = span_mask
            .unsqueeze(0)
            .expand(&[num_schemas, seq_len, max_width], true);
        let masked_scores = scores.where_self(&mask_expanded, &Tensor::ones(&[1], (Kind::Float, self.device)) * -1e9);

        Ok(masked_scores)
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
        let mut layer = SpanRepresentationLayer::new(self.hidden_size, self.max_width, self.device)?;

        if self.use_layer_norm {
            // Initialize layer norm parameters
            let weight = Tensor::ones(&[self.hidden_size as i64], (Kind::Float, self.device));
            let bias = Tensor::zeros(&[self.hidden_size as i64], (Kind::Float, self.device));
            layer.layer_norm = Some(Tensor::cat(&[&weight, &bias], 0));
        }

        if self.use_linear_projection {
            // Initialize linear projection parameters
            let w = Tensor::randn(
                &[self.hidden_size as i64, (self.hidden_size * 2) as i64],
                (Kind::Float, self.device),
            ) * 0.02;
            let b = Tensor::zeros(&[self.hidden_size as i64], (Kind::Float, self.device));
            layer.span_linear_w = Some(w);
            layer.span_linear_b = Some(b);
        }

        Ok(layer)
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
        let token_embs = Tensor::randn(&[10, 768], (Kind::Float, Device::Cpu));

        let result = layer.forward(&token_embs);
        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.span_rep.size(), &[10, 8, 768]);
        assert_eq!(result.spans_idx.size(), &[10, 8, 2]);
        assert_eq!(result.span_mask.size(), &[10, 8]);
    }

    #[test]
    fn test_span_rep_forward_single_token() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(&[1, 768], (Kind::Float, Device::Cpu));

        let result = layer.forward(&token_embs);
        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.span_rep.size()[0], 1);
        assert_eq!(result.span_rep.size()[1], 8);
    }

    #[test]
    fn test_span_rep_forward_various_lengths() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();

        for length in [1, 2, 5, 10, 20, 50] {
            let token_embs = Tensor::randn(&[length, 768], (Kind::Float, Device::Cpu));
            let result = layer.forward(&token_embs);
            assert!(result.is_ok());
            let result = result.unwrap();
            assert_eq!(result.span_rep.size(), &[length, 8, 768]);
        }
    }

    #[test]
    fn test_span_rep_batch() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let lengths = [3, 7, 12, 5, 1];
        let embs_list: Vec<Tensor> = lengths
            .iter()
            .map(|&l| Tensor::randn(&[l, 768], (Kind::Float, Device::Cpu)))
            .collect();

        let results = layer.forward_batch(&embs_list);
        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 5);

        for (i, (result, &length)) in results.iter().zip(lengths.iter()).enumerate() {
            assert_eq!(result.span_rep.size()[0], length, "Sample {} length mismatch", i);
        }
    }

    #[test]
    fn test_span_rep_invalid_input() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();

        // Wrong number of dimensions
        let bad_embs = Tensor::randn(&[10], (Kind::Float, Device::Cpu));
        assert!(layer.forward(&bad_embs).is_err());

        // Wrong hidden size
        let bad_embs = Tensor::randn(&[10, 512], (Kind::Float, Device::Cpu));
        assert!(layer.forward(&bad_embs).is_err());
    }

    #[test]
    fn test_span_rep_builder() {
        let layer = SpanRepBuilder::new()
            .hidden_size(512)
            .max_width(12)
            .use_layer_norm(true)
            .use_linear_projection(false)
            .device(Device::Cpu)
            .build();

        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 512);
        assert_eq!(layer.max_width, 12);
        assert!(layer.layer_norm.is_some());
        assert!(layer.span_linear_w.is_none());
    }

    #[test]
    fn test_span_rep_with_linear_projection() {
        let layer = SpanRepBuilder::new()
            .hidden_size(768)
            .max_width(8)
            .use_linear_projection(true)
            .build()
            .unwrap();

        let token_embs = Tensor::randn(&[10, 768], (Kind::Float, Device::Cpu));
        let result = layer.forward(&token_embs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_span_mask_correctness() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(&[5, 768], (Kind::Float, Device::Cpu));

        let result = layer.forward(&token_embs).unwrap();
        let mask = result.span_mask;

        // For seq_len=5, max_width=8:
        // Width 0: all 5 positions valid
        // Width 1: positions 0-3 valid (end < 5)
        // Width 2: positions 0-2 valid
        // Width 3: positions 0-1 valid
        // Width 4: position 0 valid
        // Width 5-7: no valid positions

        let mask_data: Vec<i64> = mask.into();
        let mut expected = vec![0i64; 5 * 8];

        for i in 0..5 {
            for w in 0..8 {
                let idx = i * 8 + w;
                expected[idx] = if i + w < 5 { 1 } else { 0 };
            }
        }

        assert_eq!(mask_data, expected);
    }

    #[test]
    fn test_span_rep_deterministic() {
        let layer = SpanRepresentationLayer::new(768, 8, Device::Cpu).unwrap();
        let token_embs = Tensor::randn(&[10, 768], (Kind::Float, Device::Cpu));

        let r1 = layer.forward(&token_embs).unwrap();
        let r2 = layer.forward(&token_embs).unwrap();

        assert_eq!(r1.span_rep, r2.span_rep);
        assert_eq!(r1.spans_idx, r2.spans_idx);
        assert_eq!(r1.span_mask, r2.span_mask);
    }
}
