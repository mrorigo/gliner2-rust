#!/usr/bin/env python3
"""
Offline Python reference parity output.

Usage:
  .venv/bin/python debug_comparison/python_parity_reference.py /abs/path/to/snapshot
"""

from __future__ import annotations

import json
import sys
from typing import Any

from gliner2 import GLiNER2


def _norm_text(v: Any) -> str:
    return str(v).strip().lower()


def normalize_entities(result: dict[str, Any]) -> dict[str, list[str]]:
    entities = result.get("entities", {})
    out: dict[str, list[str]] = {}
    if not isinstance(entities, dict):
        return out
    for key, values in entities.items():
        if isinstance(values, list):
            out[key] = sorted({_norm_text(v.get("text") if isinstance(v, dict) else v) for v in values if v})
        else:
            out[key] = []
    return out


def normalize_relations(result: dict[str, Any]) -> dict[str, list[str]]:
    rel = result.get("relation_extraction", {})
    out: dict[str, list[str]] = {}
    if not isinstance(rel, dict):
        return out
    for name, values in rel.items():
        pairs: list[str] = []
        if isinstance(values, list):
            for item in values:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append(f"{_norm_text(item[0])}|||{_norm_text(item[1])}")
                elif isinstance(item, dict):
                    head = item.get("head")
                    tail = item.get("tail")
                    if isinstance(head, dict):
                        head = head.get("text")
                    if isinstance(tail, dict):
                        tail = tail.get("text")
                    if head and tail:
                        pairs.append(f"{_norm_text(head)}|||{_norm_text(tail)}")
        out[name] = sorted(set(pairs))
    return out


def normalize_structures(result: dict[str, Any], key: str) -> list[dict[str, Any]]:
    vals = result.get(key, [])
    out: list[dict[str, Any]] = []
    if not isinstance(vals, list):
        return out
    for inst in vals:
        if not isinstance(inst, dict):
            continue
        norm_inst: dict[str, Any] = {}
        for field, value in inst.items():
            if isinstance(value, list):
                texts = []
                for v in value:
                    if isinstance(v, dict):
                        v = v.get("text")
                    if v is not None:
                        texts.append(_norm_text(v))
                norm_inst[field] = sorted(set(texts))
            elif isinstance(value, dict):
                text = value.get("text")
                norm_inst[field] = _norm_text(text) if text is not None else None
            elif value is None:
                norm_inst[field] = None
            else:
                norm_inst[field] = _norm_text(value)
        out.append(norm_inst)
    out.sort(key=lambda x: json.dumps(x, sort_keys=True))
    return out


def normalize_classification(result: dict[str, Any], task: str) -> Any:
    value = result.get(task)
    if isinstance(value, str):
        return _norm_text(value)
    if isinstance(value, dict):
        label = value.get("label")
        return _norm_text(label) if label is not None else None
    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("label")
            if item is not None:
                labels.append(_norm_text(item))
        return sorted(set(labels))
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python_parity_reference.py <snapshot_path>", file=sys.stderr)
        return 2

    snapshot = sys.argv[1]
    model = GLiNER2.from_pretrained(snapshot)

    fixtures = {
        "entities_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
        "class_text": "I love this product.",
        "relations_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
        "structure_text": "Apple announced iPhone in Cupertino, California.",
    }

    entities_raw = model.extract_entities(
        fixtures["entities_text"],
        ["person", "organization", "location"],
        include_confidence=False,
        include_spans=False,
    )
    class_raw = model.classify_text(
        fixtures["class_text"],
        {"sentiment": ["positive", "negative"]},
        include_confidence=False,
    )
    rel_raw = model.extract_relations(
        fixtures["relations_text"],
        ["works_for"],
        include_confidence=False,
        include_spans=False,
    )
    struct_schema = {
        "entities": [],
        "classifications": [],
        "relations": [],
        "json_structures": [{"product_info": {"name": "", "company": ""}}],
    }
    struct_raw = model.extract(
        fixtures["structure_text"],
        struct_schema,
        include_confidence=False,
        include_spans=False,
    )

    result = {
        "entities": normalize_entities(entities_raw),
        "classification": normalize_classification(class_raw, "sentiment"),
        "relations": normalize_relations(rel_raw),
        "structures": normalize_structures(struct_raw, "product_info"),
    }

    print("__PARITY_JSON_START__")
    print(json.dumps(result, sort_keys=True))
    print("__PARITY_JSON_END__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
