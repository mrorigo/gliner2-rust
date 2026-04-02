//! DeBERTa V3 encoder for GLiNER2.
//!
//! This implements the exact DeBERTa V3 architecture used by GLiNER2:
//! - Disentangled multi-head attention with relative position bias (c2p + p2c)
//! - query_proj, key_proj, value_proj naming
//! - Relative position embeddings (rel_embeddings)
//! - No token_type_embeddings

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

#[derive(Clone)]
pub struct DebertaV3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub max_relative_positions: isize,
    pub pos_att_type: Vec<String>,
    pub position_buckets: usize,
    pub share_att_key: bool,
    pub relative_attention: bool,
}

impl Default for DebertaV3Config {
    fn default() -> Self {
        Self {
            vocab_size: 128011,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            position_buckets: 256,
            share_att_key: true,
            relative_attention: true,
        }
    }
}

/// DeBERTa V3 embeddings (word_embeddings + LayerNorm, no token_type_embeddings)
struct DebertaV3Embeddings {
    word_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl DebertaV3Embeddings {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self { word_embeddings, layer_norm })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.word_embeddings.forward(input_ids)?;
        self.layer_norm.forward(&embeddings)
    }
}

/// Self-attention with disentangled relative position bias (DeBERTa V3 style)
struct DebertaV3Attention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_dense: Linear,
    output_layer_norm: LayerNorm,
    num_attention_heads: usize,
    attention_head_size: usize,
    pos_dropout_prob: f64,
    max_relative_positions: isize,
    position_buckets: usize,
    share_att_key: bool,
    pos_att_type: Vec<String>,
}

impl DebertaV3Attention {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let vb_self = vb.pp("attention").pp("self");
        let vb_out = vb.pp("attention").pp("output");
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let query_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb_self.pp("query_proj"))?;
        let key_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb_self.pp("key_proj"))?;
        let value_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb_self.pp("value_proj"))?;

        let output_dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb_out.pp("dense"))?;
        let output_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb_out.pp("LayerNorm"),
        )?;

        Ok(Self {
            query_proj, key_proj, value_proj,
            output_dense, output_layer_norm,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            pos_dropout_prob: config.hidden_dropout_prob,
            max_relative_positions: config.max_relative_positions,
            position_buckets: config.position_buckets,
            share_att_key: config.share_att_key,
            pos_att_type: config.pos_att_type.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, rel_embeddings: Option<&Tensor>) -> Result<Tensor> {
        let input_tensor = hidden_states.clone();
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;

        // Project Q, K, V
        let query_states = self.query_proj.forward(hidden_states)?;
        let key_states = self.key_proj.forward(hidden_states)?;
        let value_states = self.value_proj.forward(hidden_states)?;

        // Reshape for multi-head attention: (batch, seq, heads, head_size) -> (batch, heads, seq, head_size)
        let query_layer = query_states.reshape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size))?.transpose(1, 2)?;
        let key_layer = key_states.reshape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size))?.transpose(1, 2)?;
        let value_layer = value_states.reshape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size))?.transpose(1, 2)?;

        // Compute scale factor based on pos_att_type
        let mut scale_factor = 1.0f64;
        if self.pos_att_type.iter().any(|s| s == "c2p") {
            scale_factor += 1.0;
        }
        if self.pos_att_type.iter().any(|s| s == "p2c") {
            scale_factor += 1.0;
        }
        let scale = (self.attention_head_size as f64 * scale_factor).sqrt();

        // Content-based attention scores: (batch, heads, seq, head_size) @ (batch, heads, head_size, seq)
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        attention_scores = (attention_scores / scale)?;

        // Add disentangled attention bias if relative embeddings are available
        if let Some(rel_emb) = rel_embeddings {
            let rel_att = self.disentangled_attention_bias(
                &query_layer, &key_layer, rel_emb, scale, batch_size, seq_len,
            )?;
            attention_scores = attention_scores.add(&rel_att)?;
        }

        // Apply attention mask
        attention_scores = attention_scores.add(&attention_mask)?;
        let attention_probs = candle_nn::ops::softmax(&attention_scores, 3)?;

        // Context layer: (batch, heads, seq, seq) @ (batch, heads, seq, head_size)
        let context = attention_probs.matmul(&value_layer)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        let context = context.reshape((batch_size, seq_len, hidden_size))?;

        // Output projection + residual + layer norm
        let output = self.output_dense.forward(&context)?;
        let output = output.add(&input_tensor)?;
        self.output_layer_norm.forward(&output)
    }

    /// Compute disentangled attention bias with c2p and p2c components.
    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        rel_embeddings: &Tensor,
        scale: f64,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Apply dropout to relative embeddings (simplified: skip dropout in inference)
        let rel_embeddings = rel_embeddings.clone();

        // Build relative position matrix: (1, seq, seq)
        let relative_pos = build_relative_position(
            seq_len, seq_len, self.position_buckets, self.max_relative_positions, query_layer.device(),
        )?;

        let mut score = Tensor::zeros(
            (batch_size, self.num_attention_heads, seq_len, seq_len),
            DType::F32,
            query_layer.device(),
        )?;

        // Content-to-Position (c2p): query @ pos_key^T
        if self.pos_att_type.iter().any(|s| s == "c2p") {
            let c2p_att = self.content_to_position(
                query_layer, &rel_embeddings, &relative_pos, scale, batch_size, seq_len,
            )?;
            score = score.add(&c2p_att)?;
        }

        // Position-to-Content (p2c): key @ pos_query^T
        if self.pos_att_type.iter().any(|s| s == "p2c") {
            let p2c_att = self.position_to_content(
                query_layer, key_layer, &rel_embeddings, &relative_pos, scale, batch_size, seq_len,
            )?;
            score = score.add(&p2c_att)?;
        }

        Ok(score)
    }

    /// Content-to-Position attention: query @ pos_key^T, gathered by relative positions
    fn content_to_position(
        &self,
        query_layer: &Tensor,
        rel_embeddings: &Tensor,
        relative_pos: &Tensor,
        scale: f64,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Compute position key projections
        let pos_key_layer = if self.share_att_key {
            // Reuse key_proj weights
            let pos_key = self.key_proj.forward(rel_embeddings)?;
            pos_key.reshape((rel_embeddings.dims()[0], self.num_attention_heads, self.attention_head_size))?
                .transpose(0, 1)?
        } else {
            candle_core::bail!("share_att_key=false not implemented")
        };

        // Repeat for batch size: (heads, pos_emb_size, head_size) -> (batch, heads, pos_emb_size, head_size)
        let pos_key_layer = pos_key_layer.unsqueeze(0)?.broadcast_as((
            batch_size, self.num_attention_heads, pos_key_layer.dims()[1], self.attention_head_size,
        ))?;

        // query @ pos_key^T: (batch, heads, seq, head_size) @ (batch, heads, head_size, pos_emb_size)
        let c2p_att = query_layer.matmul(&pos_key_layer.transpose(2, 3)?)?;

        // Gather using relative positions
        let att_span = rel_embeddings.dims()[0] / 2;
        let att_span_tensor = Tensor::full(att_span as f32, relative_pos.dims(), relative_pos.device())?;
        let shifted_pos = relative_pos.add(&att_span_tensor)?;
        let max_val = (att_span * 2 - 1) as f32;
        let shifted_pos = shifted_pos.clamp(0.0, max_val)?;

        // Gather along last dim
        let gathered = gather_along_last_dim(&c2p_att, &shifted_pos)?;

        Ok((gathered / scale)?)
    }

    /// Position-to-Content attention: key @ pos_query^T, gathered by relative positions
    fn position_to_content(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        rel_embeddings: &Tensor,
        relative_pos: &Tensor,
        scale: f64,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Compute position query projections
        let pos_query_layer = if self.share_att_key {
            // Reuse query_proj weights
            let pos_query = self.query_proj.forward(rel_embeddings)?;
            pos_query.reshape((rel_embeddings.dims()[0], self.num_attention_heads, self.attention_head_size))?
                .transpose(0, 1)?
        } else {
            candle_core::bail!("share_att_key=false not implemented")
        };

        // Repeat for batch size
        let pos_query_layer = pos_query_layer.unsqueeze(0)?.broadcast_as((
            batch_size, self.num_attention_heads, pos_query_layer.dims()[1], self.attention_head_size,
        ))?;

        // key @ pos_query^T: (batch, heads, seq, head_size) @ (batch, heads, head_size, pos_emb_size)
        let p2c_att = key_layer.matmul(&pos_query_layer.transpose(2, 3)?)?;

        // For p2c, we need to gather with negated relative positions
        let att_span = rel_embeddings.dims()[0] / 2;
        let neg_one = Tensor::full(-1.0f32, relative_pos.dims(), relative_pos.device())?;
        let neg_rel_pos = relative_pos.mul(&neg_one)?;
        let att_span_tensor = Tensor::full(att_span as f32, relative_pos.dims(), relative_pos.device())?;
        let shifted_pos = neg_rel_pos.add(&att_span_tensor)?;
        let max_val = (att_span * 2 - 1) as f32;
        let shifted_pos = shifted_pos.clamp(0.0, max_val)?;

        // Gather along last dim
        let gathered = gather_along_last_dim(&p2c_att, &shifted_pos)?;

        Ok((gathered / scale)?)
    }
}

/// Feed-forward layer
struct DebertaV3Intermediate {
    dense: Linear,
    output_dense: Linear,
    output_layer_norm: LayerNorm,
}

impl DebertaV3Intermediate {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let vb_int = vb.pp("intermediate");
        let vb_out = vb.pp("output");
        let dense = candle_nn::linear(config.hidden_size, config.intermediate_size, vb_int.pp("dense"))?;
        let output_dense = candle_nn::linear(config.intermediate_size, config.hidden_size, vb_out.pp("dense"))?;
        let output_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb_out.pp("LayerNorm"),
        )?;

        Ok(Self { dense, output_dense, output_layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_tensor = hidden_states.clone();
        let hidden = self.dense.forward(hidden_states)?;
        let hidden = hidden.gelu()?;
        let output = self.output_dense.forward(&hidden)?;
        let output = output.add(&input_tensor)?;
        self.output_layer_norm.forward(&output)
    }
}

/// DeBERTa V3 layer
struct DebertaV3Layer {
    attention: DebertaV3Attention,
    intermediate: DebertaV3Intermediate,
}

impl DebertaV3Layer {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let attention = DebertaV3Attention::load(vb.clone(), config)?;
        let intermediate = DebertaV3Intermediate::load(vb, config)?;
        Ok(Self { attention, intermediate })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, rel_embeddings: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.attention.forward(hidden_states, attention_mask, rel_embeddings)?;
        self.intermediate.forward(&hidden)
    }
}

/// DeBERTa V3 encoder
struct DebertaV3Encoder {
    layers: Vec<DebertaV3Layer>,
    final_layer_norm: LayerNorm,
    num_attention_heads: usize,
}

impl DebertaV3Encoder {
    fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DebertaV3Layer::load(vb.pp("layer").pp(i.to_string()), config)?;
            layers.push(layer);
        }
        let final_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self { layers, final_layer_norm, num_attention_heads: config.num_attention_heads })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, rel_embeddings: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, attention_mask, rel_embeddings)?;
        }
        self.final_layer_norm.forward(&hidden)
    }
}

/// DeBERTa V3 model (matches GLiNER2's encoder architecture)
pub struct DebertaV3Model {
    embeddings: DebertaV3Embeddings,
    encoder: DebertaV3Encoder,
    rel_embeddings: Option<Embedding>,
    device: Device,
}

impl DebertaV3Model {
    pub fn load(vb: VarBuilder, config: &DebertaV3Config) -> Result<Self> {
        let embeddings = DebertaV3Embeddings::load(vb.pp("embeddings"), config)?;
        let encoder = DebertaV3Encoder::load(vb.pp("encoder"), config)?;

        // Load relative position embeddings from encoder.rel_embeddings
        let rel_embeddings = if vb.contains_tensor("encoder.rel_embeddings.weight") {
            Some(candle_nn::embedding(
                512,
                config.hidden_size,
                vb.pp("encoder").pp("rel_embeddings"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            rel_embeddings,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        _token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;

        let attention_mask = match attention_mask {
            Some(mask) => mask.clone(),
            None => input_ids.ones_like()?,
        };

        // Create extended attention mask for multi-head attention
        // Input: (batch, seq_len) with 1s for valid tokens, 0s for padding
        // Output: (batch, heads, seq_len, seq_len) for broadcasting with (batch, heads, seq, seq)
        let attention_mask = match attention_mask.rank() {
            2 => {
                // (batch, seq) -> (batch, 1, 1, seq) -> broadcast to (batch, heads, seq, seq)
                let mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;
                let seq_len = mask.dims()[3];
                let num_heads = self.encoder.num_attention_heads;
                // Broadcast to (batch, heads, seq, seq)
                mask.broadcast_as((mask.dims()[0], num_heads, seq_len, seq_len))?
            }
            _ => candle_core::bail!("Wrong shape for attention_mask"),
        };
        let attention_mask = attention_mask.to_dtype(DType::F32)?;
        // Invert mask: 0 for valid, -inf for padding
        let attention_mask = (attention_mask.ones_like()? - &attention_mask)?
            .broadcast_mul(&Tensor::try_from(f32::MIN)?.to_device(attention_mask.device())?)?;

        let rel_embeddings = self.rel_embeddings.as_ref().map(|e| e.embeddings());
        self.encoder.forward(&embedding_output, &attention_mask, rel_embeddings.as_ref().map(|t| *t))
    }

    pub fn device(&self) -> &Device { &self.device }
}

/// Build relative position matrix: rel_pos[i,j] = i - j
/// Returns tensor of shape (1, query_size, key_size)
fn build_relative_position(
    query_size: usize,
    key_size: usize,
    position_buckets: usize,
    max_relative_positions: isize,
    device: &Device,
) -> Result<Tensor> {
    // Create position indices
    let q_ids: Vec<f32> = (0..query_size).map(|i| i as f32).collect();
    let k_ids: Vec<f32> = (0..key_size).map(|i| i as f32).collect();

    let q_tensor = Tensor::from_slice(&q_ids, (query_size,), device)?;
    let k_tensor = Tensor::from_slice(&k_ids, (key_size,), device)?;

    // rel_pos[i,j] = q_ids[i] - k_ids[j]
    let q_expanded = q_tensor.reshape((query_size, 1))?;
    let k_expanded = k_tensor.reshape((1, key_size))?;
    let rel_pos = q_expanded.broadcast_sub(&k_expanded)?;

    // Apply bucketing if configured
    if position_buckets > 0 && max_relative_positions > 0 {
        return make_log_bucket_position(&rel_pos, position_buckets, max_relative_positions as usize)?
            .reshape((1, query_size, key_size));
    }

    rel_pos.reshape((1, query_size, key_size))
}

/// Apply log bucketing to relative positions
fn make_log_bucket_position(
    rel_pos: &Tensor,
    bucket_size: usize,
    max_position: usize,
) -> Result<Tensor> {
    let bucket_size_f = bucket_size as f32;
    let max_position_f = max_position as f32;

    let abs_rel_pos = rel_pos.abs()?;

    // Compute sign: 1.0 if >= 0, -1.0 if < 0
    let zero = Tensor::zeros(rel_pos.dims(), DType::F32, rel_pos.device())?;
    let sign = rel_pos.ge(&zero)?.to_dtype(DType::F32)?;
    let two = Tensor::full(2.0f32, rel_pos.dims(), rel_pos.device())?;
    let one = Tensor::ones(rel_pos.dims(), DType::F32, rel_pos.device())?;
    let sign = sign.mul(&two)?.sub(&one)?;

    // For positions within bucket_size, keep as-is
    // For larger positions, apply log bucketing
    let clamped = abs_rel_pos.maximum(&one)?.minimum(&Tensor::full(max_position_f, rel_pos.dims(), rel_pos.device())?)?;

    // log2(x) = ln(x) / ln(2)
    let ln2 = (2.0f32).ln();
    let log_clamped = clamped.log()?;
    let log_bucket_size = bucket_size_f.ln();
    let ln2_tensor = Tensor::full(ln2, rel_pos.dims(), rel_pos.device())?;
    let log_bucket = log_clamped.sub(&Tensor::full(log_bucket_size, rel_pos.dims(), rel_pos.device())?)?.div(&ln2_tensor)?.floor()?;
    let bucketed = (Tensor::full(bucket_size_f, rel_pos.dims(), rel_pos.device())?.add(&log_bucket)?)
        .clamp(0.0, (bucket_size * 2 - 1) as f32)?;

    // Use original values for small positions, bucketed for large
    let bucket_size_tensor = Tensor::full(bucket_size_f, rel_pos.dims(), rel_pos.device())?;
    let small_mask = abs_rel_pos.lt(&bucket_size_tensor)?;
    let result = small_mask.where_cond(&abs_rel_pos, &bucketed)?;

    Ok(sign.mul(&result)?)
}

/// Gather value along the last dimension using indices
/// input: (batch, heads, seq, vocab_size)
/// indices: (1, seq, seq) with values in [0, vocab_size)
/// output: (batch, heads, seq, seq)
fn gather_along_last_dim(input: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let input_dims = input.dims();
    let batch_size = input_dims[0];
    let num_heads = input_dims[1];
    let seq_len = input_dims[2];
    let vocab_size = input_dims[3];

    // indices shape: (1, seq, seq) -> expand to (batch, heads, seq, seq)
    let indices_expanded = indices.broadcast_as((batch_size, num_heads, seq_len, seq_len))?;
    let indices_i64 = indices_expanded.to_dtype(DType::I64)?;
    let indices_vec: Vec<i64> = indices_i64.flatten_all()?.to_vec1()?;

    // Get input data
    let input_data: Vec<f32> = input.flatten_all()?.to_vec1()?;

    // Gather manually
    let mut result_data = Vec::with_capacity(batch_size * num_heads * seq_len * seq_len);
    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = indices_vec[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j];
                    let idx = idx as usize;
                    if idx < vocab_size {
                        let val = input_data[b * num_heads * seq_len * vocab_size + h * seq_len * vocab_size + i * vocab_size + idx];
                        result_data.push(val);
                    } else {
                        result_data.push(0.0);
                    }
                }
            }
        }
    }

    Tensor::from_slice(&result_data, (batch_size, num_heads, seq_len, seq_len), input.device())
}
