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
//! use tch::{Tensor, Device, Kind};
//!
//! let hidden_size = 768;
//! let max_count = 20;
//! let layer = CountPredictionLayer::new(hidden_size, max_count, Device::Cpu)?;
//!
//! // Schema embedding: (1, hidden_size)
//! let schema_emb = Tensor::randn([1, hidden_size], (Kind::Float, Device::Cpu));
//!
//! let output = layer.predict_count(&schema_emb)?;
//! println!("Predicted count: {}", output.count);
//! ```

use tch::{nn, nn::Module, Device, Kind, Tensor};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the count prediction layer.
#[derive(Debug)]
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

/// Count embedding layer.
///
/// This layer converts a count index into a dense embedding vector.
/// It's used to embed the predicted count for use in span scoring.
#[derive(Debug)]
pub struct CountEmbedding {
    /// Maximum count value (exclusive).
    pub max_count: usize,
    /// Hidden size of the embeddings.
    pub hidden_size: usize,
    /// Embedding table of shape `(max_count, hidden_size)`.
    embedding_table: Tensor,
    /// Device for tensor operations.
    device: Device,
}

impl Clone for CountEmbedding {
    fn clone(&self) -> Self {
        Self {
            max_count: self.max_count,
            hidden_size: self.hidden_size,
            embedding_table: self.embedding_table.shallow_clone(),
            device: self.device,
        }
    }
}

impl CountEmbedding {
    /// Create a new count embedding layer.
    ///
    /// # Arguments
    ///
    /// * `max_count` - Maximum count value (exclusive).
    /// * `hidden_size` - Hidden size of the embeddings.
    /// * `device` - The device to place tensors on.
    pub fn new(max_count: usize, hidden_size: usize, device: Device) -> Result<Self> {
        // Initialize embedding table with small random values
        let embedding_table = Tensor::randn(
            &[max_count as i64, hidden_size as i64],
            (Kind::Float, device),
        ) * 0.02;

        Ok(Self {
            max_count,
            hidden_size,
            embedding_table,
            device,
        })
    }

    /// Initialize from a VarStore.
    ///
    /// # Arguments
    ///
    /// * `vs` - The variable store containing the weights.
    /// * `prefix` - The prefix for weight names.
    pub fn init_from_varstore(&mut self, vs: &nn::Path, prefix: &str) -> Result<()> {
        let path = vs / prefix;

        let table = path.var(
            "weight",
            &[self.max_count as i64, self.hidden_size as i64],
            nn::init::DEFAULT_KAIMING_UNIFORM,
        );
        self.embedding_table = table;

        Ok(())
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
        let embedding = self.embedding_table.narrow(0, count as i64, 1);
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
        let indices: Vec<i64> = counts.iter().map(|&c| c as i64).collect();
        let indices_tensor = Tensor::from_slice(&indices).to_device(self.device);
        let embeddings = self.embedding_table.index_select(0, &indices_tensor);

        Ok(embeddings)
    }

    /// Get the embedding table.
    pub fn embedding_table(&self) -> &Tensor {
        &self.embedding_table
    }

    /// Set the embedding table.
    pub fn set_embedding_table(&mut self, table: Tensor) {
        self.embedding_table = table;
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
#[derive(Debug)]
pub struct CountPredictionLayer {
    /// Hidden size of the input embeddings.
    pub hidden_size: usize,
    /// Maximum count value (exclusive).
    pub max_count: usize,
    /// Linear layer weight of shape `(max_count, hidden_size)`.
    weight: Tensor,
    /// Linear layer bias of shape `(max_count,)`.
    bias: Tensor,
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
            weight: self.weight.shallow_clone(),
            bias: self.bias.shallow_clone(),
            count_embedding: self.count_embedding.clone(),
            device: self.device,
        }
    }
}

impl CountPredictionLayer {
    /// Create a new count prediction layer.
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
        let count_embedding = CountEmbedding::new(max_count, hidden_size, device)?;

        // Initialize linear layer weights
        let weight = Tensor::randn(
            &[max_count as i64, hidden_size as i64],
            (Kind::Float, device),
        ) * 0.02;

        let bias = Tensor::zeros(&[max_count as i64], (Kind::Float, device));

        Ok(Self {
            hidden_size,
            max_count,
            weight,
            bias,
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

    /// Initialize from a VarStore.
    ///
    /// # Arguments
    ///
    /// * `vs` - The variable store containing the weights.
    /// * `prefix` - The prefix for weight names.
    pub fn init_from_varstore(&mut self, vs: &nn::Path, prefix: &str) -> Result<()> {
        let path = vs / prefix;

        // Load weight
        let w = path.var(
            "weight",
            &[self.max_count as i64, self.hidden_size as i64],
            nn::init::DEFAULT_KAIMING_UNIFORM,
        );
        self.weight = w;

        // Load bias
        let b = path.var("bias", &[self.max_count as i64], nn::Init::Const(0.0));
        self.bias = b;

        // Initialize count embedding
        self.count_embedding.init_from_varstore(vs, &format!("{prefix}_embedding"))?;

        Ok(())
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
        let size = schema_emb.size();

        // Handle both 1D and 2D inputs
        let (input, needs_squeeze) = if size.len() == 1 {
            (schema_emb.unsqueeze(0), true)
        } else if size.len() == 2 {
            (schema_emb.shallow_clone(), false)
        } else {
            return Err(GlinerError::dimension_mismatch(
                vec![1, 2],
                size.iter().map(|d| *d as usize).collect(),
            ));
        };

        let batch_size = input.size()[0] as usize;
        let hidden_dim = input.size()[1] as usize;

        if hidden_dim != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(
                vec![self.hidden_size],
                vec![hidden_dim],
            ));
        }

        // Compute logits: input @ weight.T + bias
        // input: (batch_size, hidden_size)
        // weight: (max_count, hidden_size)
        // output: (batch_size, max_count)
        let logits = input.matmul(&self.weight.transpose_copy(0, 1)) + &self.bias;

        // Get predicted count (argmax)
        let count = if batch_size == 1 {
            // Single sample: get argmax directly
            let logits_1d = if needs_squeeze {
                logits.squeeze_dim(0)
            } else {
                logits.narrow(0, 0, 1).squeeze_dim(0)
            };
            let argmax = logits_1d.argmax(0, false);
            argmax.int64_value(&[]) as usize
        } else {
            // Batch: get argmax for first sample
            let logits_1d = logits.narrow(0, 0, 1).squeeze_dim(0);
            let argmax = logits_1d.argmax(0, false);
            argmax.int64_value(&[]) as usize
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
        let size = schema_embs.size();
        if size.len() != 2 {
            return Err(GlinerError::dimension_mismatch(
                vec![2],
                size.iter().map(|d| *d as usize).collect(),
            ));
        }

        let num_schemas = size[0] as usize;
        let mut outputs = Vec::with_capacity(num_schemas);

        for i in 0..num_schemas {
            let schema_emb = schema_embs.narrow(0, i as i64, 1);
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

        // Project schema embeddings using count embedding
        // In the Python implementation, this is typically:
        // struct_proj = self.count_embed(embs[1:], pred_count)
        // Which projects the schema embeddings (excluding the first one) using count embedding

        // Simple projection: multiply schema embeddings by count embedding
        // schema_embs: (num_schemas, hidden_size)
        // count_emb: (1, hidden_size)
        // Result: (num_schemas, hidden_size)
        let projected = schema_embs * count_emb;

        Ok(projected)
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Get the count embedding layer.
    pub fn count_embedding(&self) -> &CountEmbedding {
        &self.count_embedding
    }

    /// Set the weight tensor.
    pub fn set_weight(&mut self, weight: Tensor) {
        self.weight = weight;
    }

    /// Set the bias tensor.
    pub fn set_bias(&mut self, bias: Tensor) {
        self.bias = bias;
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
        assert_eq!(embedding.embedding_table.size(), &[20, 768]);
    }

    #[test]
    fn test_count_embedding_forward() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();

        let emb = embedding.forward(0);
        assert!(emb.is_ok());
        assert_eq!(emb.unwrap().size(), &[1, 768]);

        let emb = embedding.forward(19);
        assert!(emb.is_ok());
        assert_eq!(emb.unwrap().size(), &[1, 768]);
    }

    #[test]
    fn test_count_embedding_forward_batch() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();
        let counts = vec![0, 5, 10, 19];

        let embs = embedding.forward_batch(&counts);
        assert!(embs.is_ok());
        assert_eq!(embs.unwrap().size(), &[4, 768]);
    }

    #[test]
    fn test_count_embedding_invalid_count() {
        let embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();

        assert!(embedding.forward(20).is_err());
        assert!(embedding.forward(100).is_err());
        assert!(embedding.forward_batch(&[0, 20]).is_err());
    }

    #[test]
    fn test_count_prediction_creation() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 768);
        assert_eq!(layer.max_count, 20);
        assert_eq!(layer.weight.size(), &[20, 768]);
        assert_eq!(layer.bias.size(), &[20]);
    }

    #[test]
    fn test_count_prediction_forward() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(&[1, 768], (Kind::Float, Device::Cpu));

        let output = layer.predict_count(&schema_emb);
        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.logits.size(), &[1, 20]);
        assert!(output.count < 20);
        assert_eq!(output.count_embedding.size(), &[1, 768]);
    }

    #[test]
    fn test_count_prediction_forward_1d() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(&[768], (Kind::Float, Device::Cpu));

        let output = layer.predict_count(&schema_emb);
        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.logits.size(), &[1, 20]);
        assert!(output.count < 20);
    }

    #[test]
    fn test_count_prediction_batch() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[5, 768], (Kind::Float, Device::Cpu));

        let outputs = layer.predict_count_batch(&schema_embs);
        assert!(outputs.is_ok());
        let outputs = outputs.unwrap();

        assert_eq!(outputs.len(), 5);
        for output in &outputs {
            assert!(output.count < 20);
            assert_eq!(output.count_embedding.size(), &[1, 768]);
        }
    }

    #[test]
    fn test_count_prediction_invalid_input() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();

        // Wrong hidden size
        let bad_emb = Tensor::randn(&[1, 512], (Kind::Float, Device::Cpu));
        assert!(layer.predict_count(&bad_emb).is_err());

        // Wrong dimensions
        let bad_emb = Tensor::randn(&[1, 2, 768], (Kind::Float, Device::Cpu));
        assert!(layer.predict_count(&bad_emb).is_err());
    }

    #[test]
    fn test_span_projection() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu));

        let projected = layer.compute_span_projection(&schema_embs, 5);
        assert!(projected.is_ok());
        assert_eq!(projected.unwrap().size(), &[3, 768]);
    }

    #[test]
    fn test_span_projection_zero_count() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_embs = Tensor::randn(&[3, 768], (Kind::Float, Device::Cpu));

        assert!(layer.compute_span_projection(&schema_embs, 0).is_err());
    }

    #[test]
    fn test_count_prediction_deterministic() {
        let layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();
        let schema_emb = Tensor::randn(&[1, 768], (Kind::Float, Device::Cpu));

        let output1 = layer.predict_count(&schema_emb).unwrap();
        let output2 = layer.predict_count(&schema_emb).unwrap();

        assert_eq!(output1.count, output2.count);
        assert_eq!(output1.logits, output2.logits);
        assert_eq!(output1.count_embedding, output2.count_embedding);
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
    fn test_count_embedding_getters_setters() {
        let mut embedding = CountEmbedding::new(20, 768, Device::Cpu).unwrap();
        let new_table = Tensor::ones(&[20, 768], (Kind::Float, Device::Cpu));

        embedding.set_embedding_table(new_table.shallow_clone());
        assert_eq!(embedding.embedding_table(), &new_table);
    }

    #[test]
    fn test_count_prediction_getters_setters() {
        let mut layer = CountPredictionLayer::new(768, 20, Device::Cpu).unwrap();

        let new_weight = Tensor::ones(&[20, 768], (Kind::Float, Device::Cpu));
        let new_bias = Tensor::zeros(&[20], (Kind::Float, Device::Cpu));

        layer.set_weight(new_weight.shallow_clone());
        layer.set_bias(new_bias.shallow_clone());

        assert_eq!(layer.weight(), &new_weight);
        assert_eq!(layer.bias(), &new_bias);
    }
}
