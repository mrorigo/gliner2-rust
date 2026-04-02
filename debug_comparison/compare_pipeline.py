"""
Focused pipeline comparison: Python vs Rust GLiNER2 implementation.
Run with: .venv/bin/python debug_comparison/compare_pipeline.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
import torch
from gliner2 import GLiNER2

print("=" * 80)
print("GLiNER2 Pipeline Comparison: Python Reference")
print("=" * 80)

# STEP 1: Load model
print("\n[STEP 1] Load model")
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
print(f"✅ Model loaded: {type(model).__name__}")

# STEP 2: Test input
TEXT = "Apple CEO Tim Cook announced new products at the headquarters in Cupertino, California."
ENTITY_TYPES = ["person", "organization", "location"]
print(f"Text: {TEXT}")
print(f"Entity types: {ENTITY_TYPES}")

# STEP 3: Tokenize
processor = model.processor
text_tokens = processor._tokenize_text(TEXT)
print(f"\n[STEP 3] text_tokens ({len(text_tokens)}): {text_tokens}")

# STEP 4: Build schema tokens
entity_tokens = ["(", "[P]", "entities", "("]
for et in ENTITY_TYPES:
    entity_tokens.append("[E]")
    entity_tokens.append(et)
entity_tokens.append(")")
entity_tokens.append(")")
print(f"\n[STEP 4] schema_tokens ({len(entity_tokens)}): {entity_tokens}")

# STEP 5: Format input with mapping (CRITICAL)
format_result = processor._format_input_with_mapping([entity_tokens], text_tokens)
input_ids = format_result['input_ids']
subwords = format_result['subword_list']
text_word_pos = format_result['text_word_first_positions']
schema_special_pos = format_result['schema_special_positions']

print(f"\n[STEP 5] Format input")
print(f"  input_ids ({len(input_ids)}): {input_ids}")
print(f"  subwords ({len(subwords)}): {subwords}")
print(f"  text_word_first_positions: {text_word_pos}")
print(f"  schema_special_positions: {schema_special_pos}")

print("\n  Tokens at schema_special_positions:")
for sidx, positions in enumerate(schema_special_pos):
    for pos in positions:
        if pos < len(subwords):
            print(f"    pos[{pos}] = '{subwords[pos]}' (id={input_ids[pos]})")

# STEP 6: Run encoder
input_ids_t = torch.tensor([input_ids], dtype=torch.long)
attn_mask = torch.ones_like(input_ids_t)

with torch.no_grad():
    encoder_out = model.encoder(
        input_ids=input_ids_t,
        attention_mask=attn_mask,
        return_dict=False
    )[0]

print(f"\n[STEP 6] Encoder output shape: {encoder_out.shape}")

# Check [E] embeddings - should be DIFFERENT
print("\n  [E] token embeddings (should be DIFFERENT):")
for pos in schema_special_pos[0]:
    if pos < encoder_out.shape[1]:
        emb = encoder_out[0, pos, :]
        print(f"    pos[{pos}] = '{subwords[pos]}' -> first 5: {emb[:5].tolist()}")

emb_4 = encoder_out[0, 4, :]
emb_6 = encoder_out[0, 6, :]
emb_8 = encoder_out[0, 8, :]
print(f"\n  Are [E] embeddings different?")
print(f"    pos[4] vs pos[6] equal: {torch.allclose(emb_4, emb_6)}")
print(f"    pos[6] vs pos[8] equal: {torch.allclose(emb_6, emb_8)}")

# STEP 7: Run full extraction
print(f"\n[STEP 7] Full entity extraction")
result = model.extract_entities(TEXT, ENTITY_TYPES, threshold=0.5, include_confidence=True, include_spans=True)
print(f"Result: {json.dumps(result, indent=2)}")

# STEP 8: Summary for Rust comparison
print(f"\n{'=' * 80}")
print("SUMMARY - Key values for Rust comparison:")
print(f"{'=' * 80}")
print(f"1. input_ids: {input_ids}")
print(f"2. text_word_first_positions: {text_word_pos}")
print(f"3. schema_special_positions: {schema_special_pos}")
print(f"4. [E] embeddings (first 5 values):")
print(f"   pos[4]: {emb_4[:5].tolist()}")
print(f"   pos[6]: {emb_6[:5].tolist()}")
print(f"   pos[8]: {emb_8[:5].tolist()}")
print(f"5. Expected: person=Tim Cook, org=Apple, loc=Cupertino+California")
