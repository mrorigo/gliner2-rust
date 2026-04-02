"""
Debug comparison script: Run the Python GLiNER2 pipeline step-by-step
and print debug output matching the Rust implementation.

Uses the installed gliner2 package (not local GLiNER2 directory).
"""
import torch
import json
from gliner2 import GLiNER2

# Same test input as Rust
TEXT = "Apple CEO Tim Cook announced new products at the headquarters in Cupertino, California."
ENTITY_TYPES = ["person", "organization", "location"]

print("=" * 80)
print("STEP 1: Load model")
print("=" * 80)
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
print(f"Model loaded: {type(model)}")

print("\n" + "=" * 80)
print("STEP 2: Build schema (same as Rust)")
print("=" * 80)
# Build schema the same way Rust does it
schema = {
    "entities": ENTITY_TYPES,
    "classifications": [],
    "relations": [],
    "json_structures": []
}
print(f"Schema: {json.dumps(schema, indent=2)}")

print("\n" + "=" * 80)
print("STEP 3: Run extract_entities with debug")
print("=" * 80)

# Let's trace through the internal pipeline
# First, let's see what the processor does
processor = model.processor
print(f"Processor type: {type(processor)}")

# Tokenize text
text_tokens = processor._tokenize_text(TEXT)
print(f"text_tokens: {text_tokens}")
print(f"text_tokens count: {len(text_tokens)}")

# Build schema tokens - the processor builds them internally
# Schema format: ["(", "[P]", "entities", "(", "[E]", "location", "[E]", "organization", "[E]", "person", ")", ")"]
entity_tokens = ["(", "[P]", "entities", "("]
for et in ENTITY_TYPES:
    entity_tokens.append("[E]")
    entity_tokens.append(et)
entity_tokens.append(")")
entity_tokens.append(")")

print(f"\nSchema tokens: {entity_tokens}")

print("\n" + "=" * 80)
print("STEP 4: Format input with mapping (the critical step)")
print("=" * 80)

# Call the internal _format_input_with_mapping
result = processor._format_input_with_mapping([entity_tokens], text_tokens)

print(f"input_ids length: {len(result['input_ids'])}")
print(f"input_ids: {result['input_ids']}")
print(f"\nsubword_list: {result['subword_list']}")
print(f"\nmapped_indices: {result['mapped_indices']}")
print(f"\ntext_word_first_positions: {result['text_word_first_positions']}")
print(f"\nschema_special_positions: {result['schema_special_positions']}")

# Verify: what tokens are at the special positions?
print("\nTokens at schema_special_positions:")
for schema_idx, positions in enumerate(result['schema_special_positions']):
    print(f"  Schema {schema_idx}: positions={positions}")
    for pos in positions:
        if pos < len(result['subword_list']):
            print(f"    pos[{pos}] = '{result['subword_list'][pos]}' (id={result['input_ids'][pos]})")

print("\n" + "=" * 80)
print("STEP 5: Run full entity extraction")
print("=" * 80)

result = model.extract_entities(TEXT, ENTITY_TYPES, threshold=0.5, include_confidence=True, include_spans=True)
print(f"Result: {json.dumps(result, indent=2)}")

print("\n" + "=" * 80)
print("STEP 6: Debug internal pipeline")
print("=" * 80)

# Access internal processor methods
# The processor has _format_input_with_mapping which we already called
# Let's check what the model's encoder produces

# Get the internal model
extractor = model.encoder
print(f"Extractor type: {type(extractor)}")

# Run a simple forward pass with the input_ids we built from _format_input_with_mapping
format_result = processor._format_input_with_mapping([entity_tokens], text_tokens)
input_ids = torch.tensor([format_result['input_ids']], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs: {input_ids[0].tolist()}")

with torch.no_grad():
    # Get encoder output
    encoder_output = extractor(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=False
    )[0]  # last_hidden_state

print(f"Encoder output shape: {encoder_output.shape}")

# Now let's check embeddings at the special positions
special_positions = format_result['schema_special_positions'][0]
print(f"\nSchema special positions: {special_positions}")
print(f"Tokens at special positions:")
for pos in special_positions:
    token = format_result['subword_list'][pos]
    emb = encoder_output[0, pos, :]
    print(f"  pos[{pos}] = '{token}' -> first 5: {emb[:5].tolist()}")

# Check if [E] embeddings are different
print(f"\nAre [E] embeddings different?")
emb_4 = encoder_output[0, 4, :]
emb_6 = encoder_output[0, 6, :]
emb_8 = encoder_output[0, 8, :]
print(f"  pos[4] vs pos[6] equal: {torch.allclose(emb_4, emb_6)}")
print(f"  pos[6] vs pos[8] equal: {torch.allclose(emb_6, emb_8)}")
print(f"  pos[4] vs pos[8] equal: {torch.allclose(emb_4, emb_8)}")

print("\n" + "=" * 80)
print("STEP 7: Summary - Key differences to check in Rust")
print("=" * 80)
print("1. Schema token order: Python uses entity order from input list")
print("2. Subword splitting: 'cupertino' -> ['▁cup', 'er', 'tino'] (3 subwords)")
print("3. text_word_first_positions skips subword positions for continuation subwords")
print("4. schema_special_positions tracks [P] and [E] token positions in subword list")
print("5. All [E] tokens have same ID (128005) but should have different embeddings due to context")
