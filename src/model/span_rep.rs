//! Span representation layer for GLiNER2 (markerV0 mode).
//!
//! Architecture from GLiNER2 Python implementation:
//! - project_start: Linear(768, 3072) + LayerNorm(3072) + Linear(3072, 768)
//! - project_end: Linear(768, 3072) + LayerNorm(3072) + Linear(3072, 768)
//! - out_project: Linear(1536, 3072) + LayerNorm(3072) + Linear(3072, 768)
//!
//! For each token position i and span width w:
//! - start_rep = project_start(token_embs[i])
//! - end_rep = project_end(token_embs[i + w])
//! - span_rep[i, w] = out_project(concat(start_rep, end_rep))

use candle_core::{Device, DType, Tensor};
use candle_nn::{Linear, VarBuilder, Module};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Output of the span representation layer.
#[derive(Clone)]
pub struct SpanRepOutput {
    pub span_rep: Tensor,      // (seq_len, max_width, hidden_size)
    pub spans_idx: Tensor,     // (seq_len, max_width, 2)
    pub span_mask: Tensor,     // (seq_len, max_width)
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

/// markerV0 span representation layer
/// Architecture: Linear + GELU + Linear (no LayerNorm)
pub struct SpanRepresentationLayer {
    pub max_width: usize,
    pub hidden_size: usize,
    // project_start: Linear(768→3072) + GELU + Linear(3072→768)
    project_start_0: Linear,      // 768 → 3072
    project_start_1: Linear,      // 3072 → 768
    // project_end: Linear(768→3072) + GELU + Linear(3072→768)
    project_end_0: Linear,        // 768 → 3072
    project_end_1: Linear,        // 3072 → 768
    // out_project: Linear(1536→3072) + GELU + Linear(3072→768)
    out_project_0: Linear,        // 1536 → 3072
    out_project_1: Linear,        // 3072 → 768
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
            project_start_0: self.project_start_0.clone(),
            project_start_1: self.project_start_1.clone(),
            project_end_0: self.project_end_0.clone(),
            project_end_1: self.project_end_1.clone(),
            out_project_0: self.out_project_0.clone(),
            out_project_1: self.out_project_1.clone(),
            device: self.device.clone(),
        }
    }
}

impl SpanRepresentationLayer {
    pub fn new(hidden_size: usize, max_width: usize, device: Device) -> Result<Self> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::build_from_varbuilder(vb, hidden_size, max_width, device)
    }

    fn build_from_varbuilder(vb: VarBuilder, hidden_size: usize, max_width: usize, device: Device) -> Result<Self> {
        let intermediate = hidden_size * 4; // 3072 for 768 hidden
        
        // project_start: Linear(768→3072) + GELU + Linear(3072→768)
        let project_start_0 = candle_nn::linear(hidden_size, intermediate, vb.pp("project_start").pp("0"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create project_start_0: {e}")))?;
        let project_start_1 = candle_nn::linear(intermediate, hidden_size, vb.pp("project_start").pp("3"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create project_start_1: {e}")))?;
        
        // project_end: Linear(768→3072) + GELU + Linear(3072→768)
        let project_end_0 = candle_nn::linear(hidden_size, intermediate, vb.pp("project_end").pp("0"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create project_end_0: {e}")))?;
        let project_end_1 = candle_nn::linear(intermediate, hidden_size, vb.pp("project_end").pp("3"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create project_end_1: {e}")))?;
        
        // out_project: Linear(1536→3072) + GELU + Linear(3072→768)
        let out_project_0 = candle_nn::linear(hidden_size * 2, intermediate, vb.pp("out_project").pp("0"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create out_project_0: {e}")))?;
        let out_project_1 = candle_nn::linear(intermediate, hidden_size, vb.pp("out_project").pp("3"))
            .map_err(|e| GlinerError::model_loading(format!("Failed to create out_project_1: {e}")))?;

        Ok(Self {
            max_width, hidden_size,
            project_start_0, project_start_1,
            project_end_0, project_end_1,
            out_project_0, out_project_1,
            device,
        })
    }

    pub fn from_config(config: &ExtractorConfig, device: Device) -> Result<Self> {
        Self::new(config.hidden_size, config.max_width, device)
    }

    pub fn from_var_builder(vb: VarBuilder, config: &ExtractorConfig, device: Device) -> Result<Self> {
        let vb = vb.pp("span_rep").pp("span_rep_layer");
        Self::build_from_varbuilder(vb, config.hidden_size, config.max_width, device)
    }

    fn project_start(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.project_start_0.forward(x)
            .map_err(|e| GlinerError::model_loading(format!("project_start_0 failed: {e}")))?;
        let x = x.gelu()
            .map_err(|e| GlinerError::model_loading(format!("project_start gelu failed: {e}")))?;
        self.project_start_1.forward(&x)
            .map_err(|e| GlinerError::model_loading(format!("project_start_1 failed: {e}")))
    }

    fn project_end(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.project_end_0.forward(x)
            .map_err(|e| GlinerError::model_loading(format!("project_end_0 failed: {e}")))?;
        let x = x.gelu()
            .map_err(|e| GlinerError::model_loading(format!("project_end gelu failed: {e}")))?;
        self.project_end_1.forward(&x)
            .map_err(|e| GlinerError::model_loading(format!("project_end_1 failed: {e}")))
    }

    fn out_project(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.out_project_0.forward(x)
            .map_err(|e| GlinerError::model_loading(format!("out_project_0 failed: {e}")))?;
        let x = x.gelu()
            .map_err(|e| GlinerError::model_loading(format!("out_project gelu failed: {e}")))?;
        self.out_project_1.forward(&x)
            .map_err(|e| GlinerError::model_loading(format!("out_project_1 failed: {e}")))
    }

    pub fn forward(&self, token_embs: &Tensor) -> Result<SpanRepOutput> {
        let dims = token_embs.dims();
        if dims.len() != 2 {
            return Err(GlinerError::dimension_mismatch(vec![2], dims.to_vec()));
        }
        let seq_len = dims[0];
        let hidden = dims[1];
        if hidden != self.hidden_size {
            return Err(GlinerError::dimension_mismatch(vec![self.hidden_size], vec![hidden]));
        }

        let mut span_reps: Vec<Tensor> = Vec::with_capacity(self.max_width);

        for width in 0..self.max_width {
            let end_idx = seq_len.saturating_sub(width);
            if end_idx == 0 {
                let zeros = Tensor::zeros((seq_len, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to create zeros: {e}")))?;
                span_reps.push(zeros);
                continue;
            }

            let start_tokens = token_embs.narrow(0, 0, end_idx)
                .map_err(|e| GlinerError::model_loading(format!("Failed to narrow start: {e}")))?;
            let end_tokens = token_embs.narrow(0, width, end_idx)
                .map_err(|e| GlinerError::model_loading(format!("Failed to narrow end: {e}")))?;

            // Project start and end
            let start_proj = self.project_start(&start_tokens)?;
            let end_proj = self.project_end(&end_tokens)?;

            // Concat and project out
            let combined = Tensor::cat(&[&start_proj, &end_proj], 1)
                .map_err(|e| GlinerError::model_loading(format!("Failed to concat: {e}")))?;
            let span_rep = self.out_project(&combined)?;

            // Pad to seq_len if needed
            if span_rep.dims()[0] < seq_len {
                let pad_size = seq_len - span_rep.dims()[0];
                let padding = Tensor::zeros((pad_size, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to create padding: {e}")))?;
                let padded = Tensor::cat(&[&span_rep, &padding], 0)
                    .map_err(|e| GlinerError::model_loading(format!("Failed to pad: {e}")))?;
                span_reps.push(padded);
            } else {
                span_reps.push(span_rep);
            }
        }

        let span_rep = Tensor::stack(&span_reps, 1)
            .map_err(|e| GlinerError::model_loading(format!("Failed to stack: {e}")))?;

        // Create span indices and mask
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

        Ok(SpanRepOutput { span_rep, spans_idx, span_mask })
    }

    pub fn forward_batch(&self, token_embs_list: &[Tensor]) -> Result<Vec<SpanRepOutput>> {
        token_embs_list.iter().map(|embs| self.forward(embs)).collect()
    }

    pub fn device(&self) -> &Device { &self.device }
}
