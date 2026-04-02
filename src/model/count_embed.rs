//! Count embedding layer for GLiNER2.
//!
//! This implements the full count_embed layer with GRU + Transformer architecture
//! that transforms entity embeddings based on predicted count.
//!
//! Architecture:
//! - Position embeddings for count positions
//! - GRU layer for sequential processing
//! - Transformer with downscaling (768→128→768)
//! - Output projection

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

/// Count embedding layer output.
#[derive(Debug, Clone)]
pub struct CountEmbedOutput {
    /// Count-aware entity embeddings: (pred_count, num_entity_types, hidden)
    pub embeddings: Tensor,
    /// Predicted count value
    pub pred_count: usize,
}

/// Multi-head attention for the transformer.
struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
}

impl MultiHeadAttention {
    fn load(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let in_proj_weight = vb
            .get((3 * embed_dim, embed_dim), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * embed_dim, "in_proj_bias")?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
            head_dim: embed_dim / num_heads,
            embed_dim,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (seq_len, batch_size, _) = x.dims3()?;

        // Q, K, V projection
        let qkv = x.matmul(&self.in_proj_weight.t()?)?;
        let qkv = qkv.add(&self.in_proj_bias.broadcast_as((batch_size, seq_len, 3 * self.embed_dim))?)?;

        // Split into Q, K, V
        let q = qkv.narrow(2, 0, self.embed_dim)?;
        let k = qkv.narrow(2, self.embed_dim, self.embed_dim)?;
        let v = qkv.narrow(2, 2 * self.embed_dim, self.embed_dim)?;

        // Reshape for multi-head attention
        let q = q.reshape((seq_len, batch_size, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .transpose(1, 2)?;
        let k = k.reshape((seq_len, batch_size, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .transpose(1, 2)?;
        let v = v.reshape((seq_len, batch_size, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .transpose(1, 2)?;

        // Attention scores
        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(2, 3)?)?;
        let attn_scores = (attn_scores / scale)?;

        let attn_probs = if let Some(m) = mask {
            let masked = attn_scores.add(m)?;
            candle_nn::ops::softmax(&masked, 3)?
        } else {
            candle_nn::ops::softmax(&attn_scores, 3)?
        };

        // Context
        let context = attn_probs.matmul(&v)?;
        let context = context.transpose(1, 2)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, batch_size, self.embed_dim))?;

        self.out_proj.forward(&context)
    }
}

/// Transformer encoder layer.
struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    fn load(vb: VarBuilder, d_model: usize, nhead: usize, dim_feedforward: usize) -> Result<Self> {
        let self_attn = MultiHeadAttention::load(vb.pp("self_attn"), d_model, nhead)?;
        let linear1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("linear2"))?;
        let norm1 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let attn_out = self.self_attn.forward(x, None)?;
        let x = x.add(&attn_out)?;
        let x = self.norm1.forward(&x)?;

        // Feed-forward with residual
        let ff_out = self.linear1.forward(&x)?;
        let ff_out = ff_out.gelu()?;
        let ff_out = self.linear2.forward(&ff_out)?;
        let x = x.add(&ff_out)?;
        self.norm2.forward(&x)
    }
}

/// GRU layer implementation.
struct GRULayer {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    hidden_size: usize,
}

impl GRULayer {
    fn load(vb: VarBuilder, input_size: usize, hidden_size: usize) -> Result<Self> {
        let weight_ih = vb.get((3 * hidden_size, input_size), "weight_ih_l0")?;
        let weight_hh = vb.get((3 * hidden_size, hidden_size), "weight_hh_l0")?;
        let bias_ih = vb.get(3 * hidden_size, "bias_ih_l0")?;
        let bias_hh = vb.get(3 * hidden_size, "bias_hh_l0")?;

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, h_0: Option<&Tensor>) -> Result<Tensor> {
        let (seq_len, batch_size, _) = x.dims3()?;

        let mut h = if let Some(h) = h_0 {
            h.clone()
        } else {
            Tensor::zeros((batch_size, self.hidden_size), DType::F32, x.device())?
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.narrow(0, t, 1)?.squeeze(0)?;

            // Gate computations
            let x_proj = x_t.matmul(&self.weight_ih.t()?)?;
            let h_proj = h.matmul(&self.weight_hh.t()?)?;
            let gates = x_proj.add(&h_proj)?.add(&self.bias_ih.add(&self.bias_hh)?)?;

            let chunk_size = self.hidden_size;
            let r_gate = gates.narrow(1, 0, chunk_size)?;
            let z_gate = gates.narrow(1, chunk_size, chunk_size)?;
            let n_gate = gates.narrow(1, 2 * chunk_size, chunk_size)?;

            // Sigmoid: 1 / (1 + exp(-x))
            let neg_r = r_gate.neg()?;
            let exp_neg_r = neg_r.exp()?;
            let ones = Tensor::full(1.0f32, exp_neg_r.dims(), exp_neg_r.device())?;
            let r_t = exp_neg_r.add(&ones)?.recip()?;

            let neg_z = z_gate.neg()?;
            let exp_neg_z = neg_z.exp()?;
            let ones = Tensor::full(1.0f32, exp_neg_z.dims(), exp_neg_z.device())?;
            let z_t = exp_neg_z.add(&ones)?.recip()?;

            let n_t = n_gate.tanh()?;

            let h_new = r_t.mul(&n_t)?.add(
                &z_t.ones_like()?.sub(&z_t)?.mul(&h)?
            )?;

            outputs.push(h_new.unsqueeze(0)?);
            h = h_new;
        }

        Tensor::cat(&outputs, 0)
    }
}

/// Full count embedding layer with GRU + Transformer.
pub struct CountEmbedLayer {
    pos_embedding: Embedding,
    gru: GRULayer,
    in_projector: Linear,
    transformer_layers: Vec<TransformerEncoderLayer>,
    out_projector_0: Linear,
    out_projector_2: Linear,
    out_projector_4: Linear,
    hidden_size: usize,
    device: Device,
}

impl CountEmbedLayer {
    /// Create a new randomly initialized count embedding layer.
    pub fn new(hidden_size: usize, max_count: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::from_var_builder(vb, hidden_size, max_count, device)
    }

    /// Create a count embedding layer from VarBuilder with loaded weights.
    pub fn from_var_builder(
        vb: VarBuilder,
        hidden_size: usize,
        max_count: usize,
        device: Device,
    ) -> Result<Self> {
        // Load position embeddings
        let pos_embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("pos_embedding"))?;

        // Load GRU
        let gru = GRULayer::load(vb.pp("gru"), hidden_size, hidden_size)?;

        // Load transformer components
        let in_projector = candle_nn::linear(hidden_size, 128, vb.pp("transformer").pp("in_projector"))?;

        let vb_transformer = vb.pp("transformer").pp("transformer");
        let mut transformer_layers = Vec::new();
        for i in 0..2 {
            let layer = TransformerEncoderLayer::load(
                vb_transformer.pp("layers").pp(i.to_string()),
                128,
                1,
                256,
            )?;
            transformer_layers.push(layer);
        }

        // Load output projector
        let out_projector_0 = candle_nn::linear(896, hidden_size, vb.pp("transformer").pp("out_projector").pp("0"))?;
        let out_projector_2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("transformer").pp("out_projector").pp("2"))?;
        let out_projector_4 = candle_nn::linear(hidden_size, hidden_size, vb.pp("transformer").pp("out_projector").pp("4"))?;

        Ok(Self {
            pos_embedding,
            gru,
            in_projector,
            transformer_layers,
            out_projector_0,
            out_projector_2,
            out_projector_4,
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
        let actual_count = pred_count.min(20);

        // Get position embeddings for each count position
        let pos_ids: Vec<u32> = (0..actual_count).map(|i| i as u32).collect();
        let pos_ids_tensor = Tensor::from_slice(&pos_ids, (actual_count,), &self.device)?;
        let pos_embs = self.pos_embedding.forward(&pos_ids_tensor)?;  // (count, hidden)

        // Expand entity embeddings: (num_types, hidden) -> (count, num_types, hidden)
        let entity_embs_expanded = entity_embs.unsqueeze(0)?
            .broadcast_as((actual_count, num_entity_types, self.hidden_size))?;

        // Expand position embeddings: (count, hidden) -> (count, num_types, hidden)
        let pos_embs_expanded = pos_embs.unsqueeze(1)?
            .broadcast_as((actual_count, num_entity_types, self.hidden_size))?;

        // Combine: entity_embs + pos_embs
        let combined = entity_embs_expanded.add(&pos_embs_expanded)?;

        // Reshape for GRU: (count, num_types, hidden) -> (count, batch=num_types, hidden)
        let gru_input = combined.clone();

        // Run GRU
        let gru_output = self.gru.forward(&gru_input, None)?;

        // Project down: (count, num_types, hidden) -> (count, num_types, 128)
        let projected = self.in_projector.forward(&gru_output)?;

        // Run transformer layers
        let mut x = projected;
        for layer in &self.transformer_layers {
            x = layer.forward(&x)?;
        }

        // Concatenate original and transformed: (count, num_types, 128 + hidden)
        let x_reshaped = x.reshape((actual_count, num_entity_types * 128))?;
        let combined_reshaped = combined.reshape((actual_count, num_entity_types * self.hidden_size))?;
        let concat = Tensor::cat(&[x_reshaped, combined_reshaped], 1)?;
        let concat = concat.reshape((actual_count, num_entity_types, 128 + self.hidden_size))?;

        // Output projection
        let out = self.out_projector_0.forward(&concat)?;
        let out = out.gelu()?;
        let out = self.out_projector_2.forward(&out)?;
        let out = out.gelu()?;
        let out = self.out_projector_4.forward(&out)?;

        Ok(CountEmbedOutput {
            embeddings: out,
            pred_count: actual_count,
        })
    }
}
