# GLiNER2 Rust Port - Implementation Plan

## Overview

This document outlines the plan for porting GLiNER2 inference from Python to Rust. The goal is to enable GLiNER2 inference in Rust while maintaining compatibility with the Python model's behavior and output format. Training, LoRA, and other non-inference features are out of scope.

## Core Architecture

GLiNER2 inference consists of three main stages:

1. **Text Tokenization & Schema Encoding** - Convert text and schema into token IDs
2. **Transformer Forward Pass** - Run the encoder (BERT-like) to get token embeddings
3. **Span/Classification Extraction** - Compute span representations and extract entities, relations, structures, and classifications

## Implementation Phases

### Phase 1: Foundation & Dependencies

**Goal**: Set up the Rust project structure and core dependencies.

**Dependencies**:
- `tokenizers` - HuggingFace tokenizer library (Rust)
- `candle` or `tch` (PyTorch bindings) - For neural network inference
  - **Recommendation**: Use `tch` for easier compatibility with PyTorch models, or `candle` for pure Rust
- `serde` / `serde_json` - JSON serialization/deserialization
- `regex` - Regex validators
- `ndarray` - N-dimensional arrays (if not using candle/tch tensors)
- `half` - FP16 support

**Tasks**:
- [ ] Configure `Cargo.toml` with dependencies
- [ ] Set up project structure (see File Structure section)
- [ ] Create basic error types and Result aliases
- [ ] Implement configuration loading from JSON/config files

### Phase 2: Tokenizer & Schema Encoding

**Goal**: Port the `SchemaTransformer` and text tokenization logic.

**Components**:

#### 2.1 Whitespace Tokenizer
- Port `WhitespaceTokenSplitter` regex-based tokenizer
- Handle URLs, emails, mentions, and word tokenization
- Map character positions to token indices

#### 2.2 Schema Transformer
- Port `SchemaTransformer` class
- Implement special token handling:
  - `[P]` - Prompt token
  - `[E]` - Entity token
  - `[C]` - Classification token
  - `[R]` - Relation token
  - `[DESC]` - Description token
  - `[EXAMPLE]` / `[OUTPUT]` - Few-shot example tokens
- Implement schema-to-tokens conversion for:
  - Entities
  - Classifications
  - JSON structures
  - Relations

#### 2.3 Batch Collation
- Port `PreprocessedBatch` data structure
- Implement `ExtractorCollator` for inference mode
- Handle padding, attention masks, and index mappings
- Implement `max_len` truncation support

### Phase 3: Model Architecture

**Goal**: Port the neural network components.

**Components**:

#### 3.1 Extractor Model
Port the `Extractor` class with these submodules:

- **Encoder**: Transformer encoder (BERT-like)
  - Load from HuggingFace format
  - Support `bert-base-uncased` and similar architectures
  
- **Span Representation Layer** (`span_rep`)
  - Compute span representations from token embeddings
  - Support width-based span extraction (up to `max_width`)
  - Implement both single and batched span computation
  
- **Count Prediction** (`count_embed`, `count_pred`)
  - Predict number of instances for each schema
  - Count embedding layer
  - Count prediction layer
  
- **Classifier** (`classifier`)
  - Classification head for classification tasks
  - Linear layer with configurable activation

#### 3.2 Configuration
Port `ExtractorConfig`:
```python
model_name: str
max_width: int
counting_layer: str  # "count_lstm" or similar
token_pooling: str   # "first" or similar
max_len: Optional[int]
```

#### 3.3 Model Loading
- Load weights from HuggingFace format (safetensors or pytorch_model.bin)
- Support `from_pretrained()` style loading
- Handle device placement (CPU/GPU)
- Support FP16 quantization
- Support `torch.compile` equivalent if using candle

### Phase 4: Inference Pipeline

**Goal**: Implement the main extraction logic.

**Components**:

#### 4.1 Schema Builder
Port the `Schema` class:
- Fluent API for building schemas
- Support entities, classifications, structures, relations
- Schema validation
- JSON/dict conversion

#### 4.2 Embedding Extraction
- Extract token embeddings for text and schema tokens
- Implement both loop-based and fast-path (gather-based) extraction
- Handle variable-length sequences in batches

#### 4.3 Entity Extraction
Port `_extract_entities`:
- Compute span scores via attention mechanism
- Apply threshold filtering
- Handle overlap removal
- Support `include_confidence` and `include_spans` flags
- Format output correctly

#### 4.4 Relation Extraction
Port `_extract_relations`:
- Extract head/tail pairs
- Compute relation scores
- Format with confidence/spans support

#### 4.5 Structure Extraction
Port `_extract_structures`:
- Extract structured field values
- Handle `dtype="str"` vs `dtype="list"`
- Handle choice fields
- Format output correctly

#### 4.6 Classification
Port `_extract_classification_result`:
- Single-label and multi-label classification
- Support sigmoid/softmax activation
- Apply classification threshold
- Format with confidence support

### Phase 5: Main API

**Goal**: Implement the user-facing API matching Python's `GLiNER2` class.

**Methods to implement**:
- `from_pretrained(model_name_or_path)` - Load model
- `from_api(api_key, ...)` - API client (optional, can be separate crate)
- `create_schema()` - Create schema builder
- `extract_entities(text, entity_types, ...)` - Entity extraction
- `batch_extract_entities(texts, entity_types, ...)` - Batch entity extraction
- `classify_text(text, tasks, ...)` - Text classification
- `batch_classify_text(texts, tasks, ...)` - Batch classification
- `extract_json(text, structures, ...)` - Structured data extraction
- `batch_extract_json(texts, structures, ...)` - Batch structured extraction
- `extract_relations(text, relation_types, ...)` - Relation extraction
- `batch_extract_relations(texts, relation_types, ...)` - Batch relation extraction
- `extract(text, schema, ...)` - General extraction with custom schema
- `batch_extract(texts, schema, ...)` - Batch general extraction

**Parameters to support**:
- `threshold` - Confidence threshold
- `batch_size` - Batch size for processing
- `num_workers` - Parallel preprocessing workers
- `format_results` - Format output nicely
- `include_confidence` - Include confidence scores
- `include_spans` - Include character positions
- `max_len` - Maximum token length

### Phase 6: Result Formatting & Post-Processing

**Goal**: Format results to match Python output exactly.

**Components**:
- Port `format_results` method
- Handle entity dict formatting
- Handle structure formatting
- Handle relation formatting
- Handle classification formatting
- Deduplicate and sort results
- Apply regex validators (`RegexValidator`)

### Phase 7: Testing & Validation

**Goal**: Ensure correctness and performance.

**Tasks**:
- [ ] Unit tests for tokenizer
- [ ] Unit tests for schema encoding
- [ ] Unit tests for each extraction type
- [ ] Integration tests comparing Rust vs Python output
- [ ] Batch correctness tests (batch vs single-sample)
- [ ] Performance benchmarks
- [ ] Edge case tests (empty input, long text, special characters)

## File Structure

```
gliner2-rust/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API and re-exports
│   ├── config.rs                 # ExtractorConfig
│   ├── error.rs                  # Error types
│   ├── tokenizer.rs              # Whitespace tokenizer and token mapping
│   ├── schema/
│   │   ├── mod.rs                # Schema module
│   │   ├── builder.rs            # Schema builder (fluent API)
│   │   ├── transformer.rs        # SchemaTransformer
│   │   └── types.rs              # Schema types and enums
│   ├── model/
│   │   ├── mod.rs                # Model module
│   │   ├── extractor.rs          # Main Extractor model
│   │   ├── span_rep.rs           # Span representation layer
│   │   ├── count_pred.rs         # Count prediction layers
│   │   ├── classifier.rs         # Classification head
│   │   └── loading.rs            # Model weight loading
│   ├── batch/
│   │   ├── mod.rs                # Batch module
│   │   ├── preprocessed.rs       # PreprocessedBatch
│   │   └── collator.rs           # ExtractorCollator
│   ├── inference/
│   │   ├── mod.rs                # Inference module
│   │   ├── engine.rs             # Main GLiNER2 struct
│   │   ├── entities.rs           # Entity extraction
│   │   ├── relations.rs          # Relation extraction
│   │   ├── structures.rs         # Structure extraction
│   │   ├── classification.rs     # Classification
│   │   └── formatting.rs         # Result formatting
│   └── validators/
│       ├── mod.rs                # Validators module
│       └── regex.rs              # RegexValidator
├── tests/
│   ├── test_tokenizer.rs
│   ├── test_schema.rs
│   ├── test_entities.rs
│   ├── test_relations.rs
│   ├── test_structures.rs
│   ├── test_classification.rs
│   └── test_batching.rs
└── examples/
    ├── basic_extraction.rs
    ├── batch_extraction.rs
    └── custom_schema.rs
```

## Key Technical Decisions

### 1. Tensor Backend
**Decision**: Use `tch` (PyTorch bindings) initially for easier model compatibility, with option to migrate to `candle` later.

**Rationale**: 
- Direct compatibility with PyTorch model weights
- Easier debugging (can compare with Python output)
- Mature ecosystem
- Can later switch to `candle` for pure Rust if needed

### 2. Tokenizer
**Decision**: Use `tokenizers` crate from HuggingFace.

**Rationale**:
- Official Rust port
- Compatible with Python tokenizers
- Well-maintained
- Supports BERT and similar architectures

### 3. Batch Processing
**Decision**: Implement DataLoader-style batching with configurable `num_workers`.

**Rationale**:
- Match Python API behavior
- Use Rust's threading for parallel preprocessing
- Efficient GPU utilization

### 4. Memory Management
**Decision**: Use pinned memory for GPU transfers, implement efficient tensor reuse.

**Rationale**:
- Match Python's `pin_memory=True` behavior
- Reduce allocation overhead in batch processing

### 5. Output Format
**Decision**: Match Python output format exactly for drop-in compatibility.

**Rationale**:
- Users can switch between Python and Rust seamlessly
- Easier testing and validation

## API Compatibility Matrix

| Python Feature | Rust Support | Notes |
|---------------|--------------|-------|
| `from_pretrained()` | ✅ | Load from HF hub or local path |
| `from_api()` | ⚠️ | Optional, separate crate |
| `create_schema()` | ✅ | Fluent API |
| `extract_entities()` | ✅ | Single text |
| `batch_extract_entities()` | ✅ | Batch with workers |
| `classify_text()` | ✅ | Single text |
| `batch_classify_text()` | ✅ | Batch |
| `extract_json()` | ✅ | Structured data |
| `batch_extract_json()` | ✅ | Batch |
| `extract_relations()` | ✅ | Relations |
| `batch_extract_relations()` | ✅ | Batch |
| `extract()` | ✅ | Custom schema |
| `batch_extract()` | ✅ | Batch custom |
| `include_confidence` | ✅ | All extraction types |
| `include_spans` | ✅ | All extraction types |
| `max_len` | ✅ | Truncation |
| `threshold` | ✅ | Confidence threshold |
| `RegexValidator` | ✅ | Post-processing |
| `quantize()` | ⚠️ | FP16 support |
| `compile()` | ❌ | Out of scope |
| Training | ❌ | Out of scope |
| LoRA | ❌ | Out of scope |

## Performance Targets

- **Single text inference**: < 100ms on CPU for short texts
- **Batch inference**: Linear scaling with batch size, efficient GPU utilization
- **Memory**: < 2GB for base model on CPU
- **Accuracy**: Bit-identical or near-identical (< 1e-5 difference) with Python output

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Model weight loading complexity | Use safetensors format, test with multiple models |
| Tensor operation differences | Extensive testing against Python output |
| Performance on CPU | Optimize critical paths, consider SIMD |
| Tokenizer edge cases | Match Python tokenizer exactly, test edge cases |
| Batch size variations | Test with various batch sizes, ensure correctness |

## Milestones

1. **M1**: Project setup, dependencies, basic structure (Week 1)
2. **M2**: Tokenizer and schema encoding working (Week 2)
3. **M3**: Model loading and forward pass (Week 3)
4. **M4**: Entity extraction working (Week 4)
5. **M5**: All extraction types working (Week 5)
6. **M6**: API complete, batch processing (Week 6)
7. **M7**: Testing, validation, documentation (Week 7)
8. **M8**: Performance optimization, release prep (Week 8)

## Next Steps

1. Review and approve this plan
2. Set up the Rust project structure
3. Implement Phase 1 (Foundation & Dependencies)
4. Begin Phase 2 (Tokenizer & Schema Encoding)