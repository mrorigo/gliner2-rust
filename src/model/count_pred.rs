//! Count prediction layer for GLiNER2.
//!
//! This module implements the count prediction mechanism used to determine
//! how many instances (entities, relations, structures) should be extracted
//! for each schema. It consists of two components:
//!
//! 1. **Count Embedding**: Converts a count index into a dense embedding
//! 2. **Count Prediction**: Predicts the count from schema embeddings
//!
//! # Architecture
//!
//! The count prediction works as follows:
//! 1. Schema embeddings are passed through a linear layer to produce count logits
//! 2. The argmax of the logits gives the predicted count
//! 3. The predicted count is embedded using the count embedding layer
//! 4. The count embedding is used to project schema embeddings for span scoring
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rust::model::CountPredictionLayer;
//! use candle_core::{Tensor, Device};
//!
//! let hidden_size = 768;
//! let max_count = 20;
//! let layer = CountPredictionLayer::new(hidden_size, max_count, Device::Cpu)?;
//!
//! // Schema embedding: (1, hidden_size)
//! let schema_emb = Tensor::randn(0.0f32, 1.0f32, (1, hidden_size), &Device::Cpu)?;
//!
//! let output = layer.predict_count(&schema_emb)?;
//! println!("Predicted count: {}", output.count);
//! ```

use candle_core::{Device, DType, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the count prediction layer.
#[derive(Clone)]
pub struct CountPredictionOutput {
    /// Raw count logits of shape `(1, max_count)`.
    pub logits: Tensor,
    /// Predicted count (argmax of logits).
    pub count: usize,
    /// Count embedding of shape `(1, hidden_size)`.
    pub count_embedding: Tensor,
}

impl CountPredictionOutput {
    /// Create a new count prediction output.
    pub fn new(logits: Tensor, count: usize, count_embedding: Tensor) -> Self {
        Self {
            logits,
            count,
            count_embedding,
        }
    }
}

impl std::fmt::Debug for CountPredictionOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountPredictionOutput")
            .field("count", &self.count)
            .field("logits_dims", &self.logits.dims())
            .field("count_embedding_dims", &self.count_embedding.dims())
            .finish()
    }
}

/// Count embedding layer.
///
/// This layer converts a count index into a dense embedding vector.
/// It's used to embed the predicted count for use in span scoring.
pub struct CountEmbedding {
    /// Maximum count value (exclusive).
    pub max_count: usize,
    /// Hidden size of the embeddings.
    pub hidden_size: usize,
    /// Embedding table.
    embedding: Embedding,
    /// Device for tensor operations.
    device: Device,
}

impl Clone for CountEmbedding {
    fn clone(&self) -> Self {
        Self {
            max_count: self.max_count,
            hidden_size: self.hidden_size,
            embedding: self.embedding.clone(),
            device: self.device.clone(),
        }
    }
}

impl std::fmt::Debug for CountEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountEmbedding")
            .field("max_count", &self.max_count)
            .field("hidden_size", &self.hidden_size)
            .field("device", &self.device)
            .finish()
    }
}

impl CountEmbedding {
    /// Create a new count embedding layer with random initialization.
    ///
    /// # Arguments
    ///
    /// * `max_count` - Maximum count value (exclusive).
    /// * `hidden_size` - Hidden size of the embeddings.
    /// * `device` - The device to place tensors on.
    pub fn new(max_count: usize, hidden_size: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("count_embedding"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create count embedding: {e}")))?;

        Ok(Self {
            max_count,
            hidden_size,
            embedding,
            device,
        })
    }

    /// Create a count embedding layer loaded from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `vb` - The VarBuilder containing the weights.
    /// * `max_count` - Maximum count value (exclusive).
    /// * `hidden_size` - Hidden size of the embeddings.
    /// * `device` - The device to place tensors on.
    pub fn from_var_builder(
        vb: VarBuilder,
        max_count: usize,
        hidden_size: usize,
        device: Device,
    ) -> Result<Self> {
        let embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("count_embedding"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to load count embedding: {e}")))?;

        Ok(Self {
            max_count,
            hidden_size,
            embedding,
            device,
        })
    }

    /// Forward pass: get embedding for a count index.
    ///
    /// # Arguments
    ///
    /// * `count` - The count index to embed.
    ///
    /// # Returns
    ///
    /// An embedding tensor of shape `(1, hidden_size)`.
    pub fn forward(&self, count: usize) -> Result<Tensor> {
        if count >= self.max_count {
            return Err(GlinerError::validation(format!(
                "Count {count} exceeds max_count {}",
                self.max_count
            )));
        }

        // Get embedding for the count index
        let indices = Tensor::new(&[count as u32], &self.device)
            .map_err(|e| GlinerError::model_loading(format!("Failed to create count index tensor: {e}")))?;
        let embedding = self.embedding.forward(&indices)
            .map_err(|e| GlinerError::model_loading(format!("Count embedding forward failed: {e}")))?;

        Ok(embedding)
    }

    /// Forward pass for multiple counts.
    ///
    /// # Arguments
    ///
    /// * `counts` - A slice of count indices.
    ///
    /// # Returns
    ///
    /// An embedding tensor of shape `(num_counts, hidden_size)`.
    pub fn forward_batch(&self, counts: &[usize]) -> Result<Tensor> {
        if counts.is_empty() {
            return Err(GlinerError::validation("Cannot embed empty counts list"));
        }

        for &count in counts {
            if count >= self.max_count {
                return Err(GlinerError::validation(format!(
                    "Count {count} exceeds max_count {}",
                    self.max_count
                )));
            }
        }

        // Gather embeddings for all counts
        let indices: Vec<u32> = counts.iter().map(|&c| c as u32).collect();
        let indices_tensor = Tensor::from_slice(&indices, (counts.len(),), &self.device)
            .map_err(|e| GlinerError::model_loading(format!("Failed to create indices tensor: {e}")))?;
        let embeddings = self.embedding.forward(&indices_tensor)
            .map_err(|e| GlinerError::model_loading(format!("Count embedding batch forward failed: {e}")))?;

        Ok(embeddings)
    }

    /// Get the device the embedding is on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Count prediction layer for GLiNER2.
///
/// This layer predicts how many instances should be extracted for a given
/// schema. It combines a linear prediction layer with a count embedding
/// layer to produce both the predicted count and its embedding.
///
/// # Tensor Shapes
///
/// - Input: `(batch_size, hidden_size)` or `(hidden_size,)`
/// - Output logits: `(batch_size, max_count)` or `(max_count,)`
pub struct CountPredictionLayer {
    /// Hidden size of the input embeddings.
    pub hidden_size: usize,
    /// Maximum count value (exclusive).
    pub max_count: usize,
    /// Linear layer for count prediction (hidden_size -> max_count).
    linear: Linear,
    /// Count embedding layer.
    count_embedding: CountEmbedding,
    /// Device for tensor operations.
    device: Device,
}

impl Clone for CountPredictionLayer {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            max_count: self.max_count,
            linear: self.linear.clone(),
            count_embedding: self.count_embedding.clone(),
            device: self.device.clone(),
        }
    }
}

impl std::fmt::Debug for CountPredictionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountPredictionLayer")
            .field("hidden_size", &self.hidden_size)
            .field("max_count", &self.max_count)
            .field("device", &self.device)
            .finish()
    }
}

impl CountPredictionLayer {
    /// Create a new count prediction layer with random initialization.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The hidden size of the input embeddings.
    /// * `max_count` - Maximum count value to predict (exclusive).
    /// * `device` - The device to place tensors on.
    ///
    /// # Returns
    ///
    /// A new `CountPredictionLayer`.
    pub fn new(hidden_size: usize, max_count: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let count_embedding = CountEmbedding::from_var_builder(
            vb.pp("count_embedding"),
            max_count,
            hidden_size,
            device.clone(),
        )?;

        let linear = candle_nn::linear(hidden_size, max_count, vb.pp("count_linear"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create count prediction linear layer: {e}")))?;

        Ok(Self {
            hidden_size,
            max_count,
            linear,
            count_embedding,
            device,
        })
    }

    /// Create a count prediction layer from a config.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `device` - The device to place tensors on.
    pub fn from_config(config: &ExtractorConfig, device: Device) -> Result<Self> {
        // Default max_count of 20 is reasonable for most extraction tasks
        Self::new(config.hidden_size, 20, device)
    }

    /// Create a count prediction layer loaded from a VarBuilder.
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
        let hidden_size = config.hidden_size;
        let max_count = 20; // Default max_count

        let count_embedding = CountEmbedding::from_var_builder(
            vb.pp("count_embedding"),
            max_count,
            hidden_size,
            device.clone(),
        )?;

        let linear = candle_nn::linear(hidden_size, max_count, vb.pp("count_linear"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to load count prediction weights: {e}")))?;

        Ok(Self {
            hidden_size,
            max_count,
            linear,
            count_embedding,
            device,
        })
    }

    /// Forward pass: predict count from schema embedding.
    ///
    /// # Arguments
    ///
    /// * `schema_emb` - Schema embedding of shape `(1, hidden_size)` or `(hidden_size,)`.
    ///
    /// # Returns
    ///
    /// A `CountPredictionOutput` containing logits, predicted count, and count embedding.
    pub fn predict_count(&self, schema_emb: &Tensor) -> Result<CountPredictionOutput> {
        let dims = schema_emb.dims();

        // Handle both 1D and 2D inputs
        let (input, is_1d) = if dims.is_empty() {
            return Err(GlinerError::dimension_mismatch(
                vec![1, 2],
                vec![0],
            ));
        } else if dims.len() == 1 {
            // 1D input: (hidden_size,) -> (1, hidden_size)
            let reshaped = schema_emb.reshape((1, dims[0]))
                .map_err(|e| GlinerError::model_loading(format!("Failed to reshape 1D input: {e}")))?;
            (reshaped, true)
        } else if dims.len() == 2 {
            (schema_emb.clone(), false)
        } else {
            return Err(GlinerError::dimension_mismatch(
                vec![1, 2],
                dims.to_vec(),
            ));
        };

        let batch_size = input.dims()[0];
        let hidden_dim = input.dims()[1];

        if hidden_dim != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(
                vec![self.hidden_size],
                vec![hidden_dim],
            ));
        }

        // Compute logits: linear forward -> (batch_size, max_count)
        let logits = self.linear.forward(&input)
            .map_err(|e| GlinerError::model_loading(format!("Count prediction forward failed: {e}")))?;

        // Get predicted count (argmax)
        let count = if batch_size == 1 {
            // Single sample: get argmax from (1, max_count) -> squeeze to (max_count,) -> argmax
            let logits_1d = if is_1d {
                logits.squeeze(0)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to squeeze logits: {e}")))?
            } else {
                logits.narrow(0, 0, 1)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to narrow logits: {e}")))?
                    .squeeze(0)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to squeeze logits: {e}")))?
            };
            let argmax = logits_1d.argmax(0)
                .map_err(|e| GlinerError::model_loading(format!("Failed to compute argmax: {e}")))?;
            argmax.to_scalar::<u32>()
                .map_err(|e| GlinerError::model_loading(format!("Failed to get argmax value: {e}")))? as usize
        } else {
            // Batch: get argmax for first sample
            let logits_1d = logits.narrow(0, 0, 1)
                .map_err(|e| GlinerError::model_loading(format!("Failed to narrow logits: {e}")))?
                .squeeze(0)
                .map_err(|e| GlinerError::model_loading(format!("Failed to squeeze logits: {e}")))?;
            let argmax = logits_1d.argmax(0)
                .map_err(|e| GlinerError::model_loading(format!("Failed to compute argmax: {e}")))?;
            argmax.to_scalar::<u32>()
                .map_err(|e| GlinerError::model_loading(format!("Failed to get argmax value: {e}")))? as usize
        };

        // Get count embedding
        let count_embedding = self.count_embedding.forward(count)?;

        Ok(CountPredictionOutput::new(logits, count, count_embedding))
    }

    /// Forward pass for multiple schema embeddings.
    ///
    /// # Arguments
    ///
    /// * `schema_embs` - Schema embeddings of shape `(num_schemas, hidden_size)`.
    ///
    /// # Returns
    ///
    /// A vector of `CountPredictionOutput`, one for each schema.
    pub fn predict_count_batch(&self, schema_embs: &Tensor) -> Result<Vec<CountPredictionOutput>> {
        let dims = schema_embs.dims();

        if dims.len() != 2 {
            return Err(GlinerError::dimension_mismatch(
                vec![2],
                dims.to_vec(),
            ));
        }

        let num_schemas = dims[0];
        let mut outputs = Vec::with_capacity(num_schemas);

        for i in 0..num_schemas {
            let schema_emb = schema_embs.narrow(0, i, 1)
                .map_err(|e| GlinerError::model_loading(format!("Failed to narrow schema embedding: {e}")))?;
            let output = self.predict_count(&schema_emb)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Compute span projections using count embeddings.
    ///
    /// This method is used during inference to project schema embeddings
    /// for span scoring based on the predicted count.
    ///
    /// # Arguments
    ///
    /// * `schema_embs` - Schema embeddings of shape `(num_schemas, hidden_size)`.
    /// * `pred_count` - The predicted count.
    ///
    /// # Returns
    ///
    /// Projected embeddings of shape `(pred_count, hidden_size)`.
    pub fn compute_span_projection(
        &self,
        schema_embs: &Tensor,
        pred_count: usize,
    ) -> Result<Tensor> {
        if pred_count == 0 {
            return Err(GlinerError::validation("Cannot project with count 0"));
        }

        // Get count embedding for the predicted count
        let count_emb = self.count_embedding.forward(pred_count - 1)?;

        // Simple projection: multiply schema embeddings by count embedding
        // schema_embs: (num_schemas, hidden_size)
        // count_emb: (1, hidden_size)
        // Result: (num_schemas, hidden_size)
        let projected = schema_embs.broadcast_mul(&count_emb)
            .map_err(|e| GlinerError::model_loading(format!("Failed to compute span projection: {e}")))?;

        Ok(projected)
    }

    /// Get the device the layer is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the count embedding layer.
    pub fn count_embedding(&self) -> &CountEmbedding {
        &self.count_embedding
    }
}

/// Builder for constructing `CountPredictionLayer` with custom settings.
#[derive(Debug, Clone)]
pub struct CountPredictionBuilder {
    hidden_size: usize,
    max_count: usize,
    device: Device,
}

impl Default for CountPredictionBuilder {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            max_count: 20,
            device: Device::Cpu,
        }
    }
}

impl CountPredictionBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hidden size.
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Set the maximum count.
    pub fn max_count(mut self, count: usize) -> Self {
        self.max_count = count;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Build the count prediction layer.
    pub fn build(self) -> Result<CountPredictionLayer> {
        CountPredictionLayer::new(self.hidden_size, self.max_count, self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_embedding_creation() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu);
        assert!(embedding.is_ok());
        let embedding = embedding.unwrap();
        assert_eq!(embedding.max_count, 20);
        assert_eq!(embedding.hidden_size, 768);
    }

    #[test]
    fn test_count_embedding_forward() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();

        let emb = embedding.forward(0);
        assert!(emb.is_ok());
        assert_eq!(emb.unwrap().dims(), &[1, 768]);

        let emb = embedding.forward(19);
        assert!(emb.is_ok());
        assert_eq!(emb.unwrap().dims(), &[1, 768]);
    }

    #[test]
    fn test_count_embedding_forward_batch() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();

        let embs = embedding.forward_batch(&[0, 5, 10, 19]);
        assert!(embs.is_ok());
        assert_eq!(embs.unwrap().dims(), &[4, 768]);
    }

    #[test]
    fn test_count_embedding_invalid_index() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();

        assert!(embedding.forward(20).is_err());
        assert!(embedding.forward(100).is_err());
    }

    #[test]
    fn test_count_prediction_creation() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 768);
        assert_eq!(layer.max_count, 20);
    }

    #[test]
    fn test_count_prediction_predict() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(0.0f32, 1.0f32, (1, 768), &Device::Cpu).unwrap();

        let output = layer.predict_count(&schema_emb);
        assert!(output.is_ok());
        let output = output.unwrap();

        assert!(output.count < 20);
        assert_eq!(output.logits.dims(), &[1, 20]);
        assert_eq!(output.count_embedding.dims(), &[1, 768]);
    }

    #[test]
    fn test_count_prediction_1d_input() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(0.0f32, 1.0f32, (768,), &Device::Cpu).unwrap();

        let output = layer.predict_count(&schema_emb);
        assert!(output.is_ok());
        let output = output.unwrap();

        assert!(output.count < 20);
    }

    #[test]
    fn test_count_prediction_batch() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (5, 768), &Device::Cpu).unwrap();

        let outputs = layer.predict_count_batch(&schema_embs);
        assert!(outputs.is_ok());
        let outputs = outputs.unwrap();

        assert_eq!(outputs.len(), 5);
        for output in &outputs {
            assert!(output.count < 20);
        }
    }

    #[test]
    fn test_count_prediction_invalid_hidden_size() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(0.0f32, 1.0f32, (1, 512), &Device::Cpu).unwrap();

        assert!(layer.predict_count(&schema_emb).is_err());
    }

    #[test]
    fn test_count_prediction_compute_span_projection() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap();

        let projected = layer.compute_span_projection(&schema_embs, 5);
        assert!(projected.is_ok());
        assert_eq!(projected.unwrap().dims(), &[3, 768]);
    }

    #[test]
    fn test_count_prediction_zero_count_projection() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(0.0f32, 1.0f32, (3, 768), &Device::Cpu).unwrap();

        assert!(layer.compute_span_projection(&schema_embs, 0).is_err());
    }

    #[test]
    fn test_count_prediction_builder() {
        let layer = CountPredictionBuilder::new()
            .hidden_size(512)
            .max_count(30)
            .device(Device::Cpu)
            .build();

        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 512);
        assert_eq!(layer.max_count, 30);
    }

    #[test]
    fn test_count_prediction_deterministic() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(0.0f32, 1.0f32, (1, 768), &Device::Cpu).unwrap();

        let output1 = layer.predict_count(&schema_emb).unwrap();
        let output2 = layer.predict_count(&schema_emb).unwrap();

        assert_eq!(output1.count, output2.count);
    }
}
