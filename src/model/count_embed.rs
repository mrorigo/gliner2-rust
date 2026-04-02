//! Count embedding layer for GLiNER2.
//!
//! This implements a simplified version of the count_embed layer that transforms
//! entity embeddings based on predicted count. The full Python implementation
//! uses GRU + Transformer, but this simplified version uses position embeddings
//! and linear projections.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

/// Count embedding layer output.
#[derive(Debug, Clone)]
pub struct CountEmbedOutput {
    pub embeddings: Tensor,  // (pred_count, num_entity_types, hidden)
    pub pred_count: usize,
}

/// Simplified count embedding layer.
///
/// The full Python implementation uses:
/// - Position embeddings
/// - GRU layer
/// - Transformer with downscaling
///
/// This simplified version uses:
/// - Position embeddings
/// - Linear projection
#[derive(Debug, Clone)]
pub struct CountEmbedLayer {
    pos_embedding: Embedding,
    in_projector: Linear,
    out_projector: Linear,
    hidden_size: usize,
    device: Device,
}

impl CountEmbedLayer {
    /// Create a new count embedding layer.
    pub fn new(hidden_size: usize, max_count: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Position embeddings for count positions
        let pos_embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("pos_embedding"))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create pos_embedding: {e}")))?;

        // Input projector: hidden_size -> hidden_size
        let in_projector = candle_nn::linear(hidden_size, hidden_size, vb.pp("in_projector"))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create in_projector: {e}")))?;

        // Output projector: hidden_size -> hidden_size
        let out_projector = candle_nn::linear(hidden_size, hidden_size, vb.pp("out_projector"))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create out_projector: {e}")))?;

        Ok(Self {
            pos_embedding,
            in_projector,
            out_projector,
            hidden_size,
            device,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `entity_embs` - Entity embeddings, shape (num_entity_types, hidden)
    /// * `pred_count` - Predicted number of entities
    ///
    /// # Returns
    /// Count-aware entity embeddings, shape (pred_count, num_entity_types, hidden)
    pub fn forward(&self, entity_embs: &Tensor, pred_count: usize) -> Result<CountEmbedOutput> {
        let num_entity_types = entity_embs.dims()[0];

        // Get position embeddings for each count position
        let pos_ids: Vec<u32> = (0..pred_count.min(20)).map(|i| i as u32).collect();
        let pos_ids_tensor = Tensor::from_slice(&pos_ids, (pos_ids.len(),), &self.device)?;
        let pos_embs = self.pos_embedding.forward(&pos_ids_tensor)?;  // (pred_count, hidden)

        // Expand entity embeddings: (num_types, hidden) -> (pred_count, num_types, hidden)
        let entity_embs_expanded = entity_embs.unsqueeze(0)?  // (1, num_types, hidden)
            .broadcast_as((pos_embs.dims()[0], num_entity_types, self.hidden_size))?;

        // Expand position embeddings: (pred_count, hidden) -> (pred_count, num_types, hidden)
        let pos_embs_expanded = pos_embs.unsqueeze(1)?  // (pred_count, 1, hidden)
            .broadcast_as((pos_embs.dims()[0], num_entity_types, self.hidden_size))?;

        // Combine: entity_embs + pos_embs
        let combined = entity_embs_expanded.add(&pos_embs_expanded)?;

        // Project through linear layers
        let projected = self.in_projector.forward(&combined)?;
        let projected = projected.gelu()?;
        let output = self.out_projector.forward(&projected)?;

        Ok(CountEmbedOutput {
            embeddings: output,
            pred_count,
        })
    }

    /// Create a count embedding layer from a VarBuilder with loaded weights.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder containing the loaded weights
    /// * `hidden_size` - Hidden size of the model
    /// * `max_count` - Maximum count value (typically 20)
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// A CountEmbedLayer with loaded weights
    pub fn from_var_builder(
        vb: candle_nn::VarBuilder,
        hidden_size: usize,
        max_count: usize,
        device: Device,
    ) -> Result<Self> {
        // Load position embeddings
        let pos_embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("pos_embedding"))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load pos_embedding: {e}")))?;

        // Load input projector
        let in_projector = candle_nn::linear(hidden_size, hidden_size, vb.pp("in_projector"))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load in_projector: {e}")))?;

        // Load output projector
        let out_projector = candle_nn::linear(hidden_size, hidden_size, vb.pp("out_projector"))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load out_projector: {e}")))?;

        Ok(Self {
            pos_embedding,
            in_projector,
            out_projector,
            hidden_size,
            device,
        })
    }
}
