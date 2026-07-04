#!/usr/bin/env python3
"""Probe a local-LLM JournalOS ingest design without writing wiki pages.

The probe logs each semantic stage so failures are easy to localize:
context -> raw candidates -> normalize/dedupe -> placement plan -> review.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agentic_ingest as A

PILLARS_PATH = Path(__file__).resolve().parent / "scaffold" / "pillars.json"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 1)


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    i, j = cleaned.find("{"), cleaned.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return {}
    try:
        data = json.loads(cleaned[i:j + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def json_lint_error(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    i, j = cleaned.find("{"), cleaned.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return "No complete JSON object was found."
    candidate = cleaned[i:j + 1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        start = max(0, exc.pos - 180)
        end = min(len(candidate), exc.pos + 180)
        snippet = candidate[start:end]
        return (
            f"{exc.msg} at line {exc.lineno}, column {exc.colno}, char {exc.pos}.\n"
            f"Nearby text:\n{snippet}"
        )
    return "JSON parsed, but the top-level value was not the expected object."


def json_repair_hint(text: str, lint_error: str) -> str:
    lower = text.lower()
    if '"placement_decisions"' in lower and '"uncertain"' in lower and "expecting ',' delimiter" in lint_error.lower():
        return (
            "Likely structural issue: the placement_decisions list may not be closed "
            "before the uncertain field. Ensure the decisions array ends with ] before "
            "the comma that introduces \"uncertain\"."
        )
    if "expecting ',' delimiter" in lint_error.lower():
        return "Likely structural issue: a comma, closing bracket, or closing brace is missing near the parser location."
    if "unterminated string" in lint_error.lower():
        return "Likely structural issue: a string quote is missing or an unescaped quote appears inside a string."
    return "Repair only the JSON syntax indicated by the parser error."


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def pillar_guide() -> str:
    data = json.loads(PILLARS_PATH.read_text(encoding="utf-8"))
    rows = []
    for pillar in data.get("pillars", []):
        name = pillar.get("name", "")
        if name:
            rows.append(f"- {slug(name)}: {pillar.get('purpose', '')} Boundary: {pillar.get('boundary', '')}")
    return "\n".join(rows)


def profile_path(name: str) -> str:
    clean = re.sub(r"\s+", " ", str(name or "").strip())
    if not clean:
        return ""
    return f"Wiki/social-connections/profiles/{clean.replace('/', '-').replace(' ', '-')}.md"


def alias_map(context: str) -> dict[str, str]:
    data = parse_json_object(context)
    aliases: dict[str, str] = {}
    for item in data.get("aliases", []) if isinstance(data.get("aliases"), list) else []:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias") or "").strip()
        canonical = str(item.get("canonical_name") or "").strip()
        confidence = str(item.get("confidence") or "").strip().lower()
        if alias and canonical and alias != canonical and confidence in ("high", "medium"):
            aliases[alias] = canonical
    return aliases


def canonical_name(name: str, aliases: dict[str, str]) -> str:
    clean = str(name or "").strip()
    return aliases.get(clean, clean)


def home_path(home: object) -> str:
    if not isinstance(home, dict):
        return ""
    pillar = slug(str(home.get("pillar") or ""))
    title = slug(str(home.get("page_title") or home.get("title") or ""))
    if not pillar or not title:
        return ""
    return f"Wiki/{pillar}/{title}.md"


def home_title(home: object) -> str:
    if not isinstance(home, dict):
        return ""
    return str(home.get("page_title") or home.get("title") or "")


def action_task_path(item: dict, profile_name: str) -> str:
    if item.get("memory_kind") != "task" or not profile_name:
        return ""
    fact = str(item.get("fact") or "").strip()
    if not fact:
        return ""
    first = fact.split(maxsplit=1)[0].lower().strip(".,:;!?")
    action_verbs = {
        "ask", "call", "check", "contact", "discuss", "email", "follow", "message",
        "ping", "review", "send", "share", "sync", "talk", "text",
    }
    if first not in action_verbs and not fact.lower().startswith("need to "):
        return ""
    return f"Wiki/execution-and-organization/{slug(fact)}.md"


def segment_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        pieces = [para] if len(para) <= max_chars else re.split(r"(?<=[.!?])\s+", para)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current and current_len + len(piece) + 2 > max_chars:
                segments.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece) + 2
    if current:
        segments.append("\n\n".join(current))
    return segments or [text]


def call_stage(out_dir: Path, stage: str, base_url: str, model: str, prompt: str,
               max_tokens: int, timeout: int) -> str:
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    t0 = time.perf_counter()
    content, toks = A.chat(
        base_url, model, [{"role": "user", "content": prompt}],
        max_tokens=max_tokens, reasoning_budget=0, timeout=timeout
    )
    output = content.strip()
    parsed = bool(parse_json_object(output))
    if not parsed:
        lint_error = json_lint_error(output)
        repair_hint = json_repair_hint(output, lint_error)
        repair_prompt = (
            "Your previous output was invalid JSON.\n\n"
            f"JSON parser error:\n{lint_error}\n\n"
            f"Repair hint:\n{repair_hint}\n\n"
            "Do not change any facts, labels, paths, decisions, item_ids, or wording. "
            "Only repair brackets, commas, quoting, "
            "and object/list structure. Return one valid JSON object and no markdown.\n\n"
            f"PREVIOUS OUTPUT:\n{output}\n"
        )
        (stage_dir / "repair-prompt.txt").write_text(repair_prompt, encoding="utf-8")
        repair_t0 = time.perf_counter()
        repaired, repair_toks = A.chat(
            base_url, model, [{"role": "user", "content": repair_prompt}],
            max_tokens=max_tokens, reasoning_budget=0, timeout=timeout
        )
        (stage_dir / "repair-output.json").write_text(repaired.strip().rstrip() + "\n", encoding="utf-8")
        if parse_json_object(repaired):
            output = repaired.strip()
            toks += repair_toks
            parsed = True
        (stage_dir / "repair-meta.json").write_text(
            json.dumps({"seconds": elapsed(repair_t0), "tokens": repair_toks, "parsed": bool(parse_json_object(repaired))}, indent=2) + "\n",
            encoding="utf-8",
        )
    (stage_dir / "output.json").write_text(output.rstrip() + "\n", encoding="utf-8")
    meta = {"seconds": elapsed(t0), "tokens": toks, "parsed": parsed}
    (stage_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"   {stage}: parsed={meta['parsed']} time={meta['seconds']}s")
    return output


def call_text_stage(out_dir: Path, stage: str, base_url: str, model: str, prompt: str,
                    max_tokens: int, timeout: int) -> str:
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    t0 = time.perf_counter()
    content, toks = A.chat(
        base_url, model, [{"role": "user", "content": prompt}],
        max_tokens=max_tokens, reasoning_budget=0, timeout=timeout
    )
    output = content.strip()
    (stage_dir / "output.txt").write_text(output.rstrip() + "\n", encoding="utf-8")
    meta = {"seconds": elapsed(t0), "tokens": toks, "parsed": True}
    (stage_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"   {stage}: text time={meta['seconds']}s")
    return output


def context_prompt(date: str, note_text: str) -> str:
    return (
        "Stage 1: build a whole-note context map for later ingest. Do not extract final memories yet. "
        "Output JSON only, no markdown.\n\n"
        "Schema:\n"
        "{\n"
        "  \"people\": [{\"name\":\"...\",\"evidence\":\"exact quote\",\"reason\":\"why this is a person\"}],\n"
        "  \"pets\": [{\"name\":\"...\",\"evidence\":\"exact quote\",\"reason\":\"why this is a pet\"}],\n"
        "  \"aliases\": [{\"alias\":\"short name or initial\",\"canonical_name\":\"full name\","
        "\"evidence\":[\"exact quote\",\"exact quote\"],\"confidence\":\"high|medium|low\","
        "\"reason\":\"why the note supports this alias\"}],\n"
        "  \"active_topics\": [{\"topic\":\"...\",\"evidence\":\"exact quote\",\"reason\":\"why it may matter later\"}],\n"
        "  \"subject_carryovers\": [{\"phrase\":\"...\",\"likely_subject\":\"self|name|unclear\",\"evidence\":\"exact quote\",\"reason\":\"short\"}],\n"
        "  \"durability_notes\": [{\"text\":\"...\",\"reason\":\"what kinds of facts are durable in this note\"}]\n"
        "}\n\n"
        "Keep reasons short. The journal author/self is not a named person entity. "
        "Only add aliases when this note itself supports the mapping. Do not merge names just because they share a first letter.\n\n"
        f"Date: {date}\n--- NOTE ---\n{note_text}\n--- END NOTE ---\n"
    )


def context_blocks_prompt(date: str, note_text: str) -> str:
    return (
        "Stage 1: build a whole-note context map for later ingest. Do not extract final memories yet. "
        "Use plain text blocks, not JSON or markdown.\n\n"
        "Use any of these block types when useful:\n\n"
        "PERSON\n"
        "name: person's name\n"
        "evidence: exact quote\n"
        "reason: why this is a person\n\n"
        "PET\n"
        "name: pet's name\n"
        "evidence: exact quote\n"
        "reason: why this is a pet\n\n"
        "ALIAS\n"
        "alias: short name or initial\n"
        "canonical_name: full name\n"
        "confidence: high|medium|low\n"
        "evidence: exact quote\n"
        "reason: why the note supports this alias\n\n"
        "TOPIC\n"
        "topic: active topic\n"
        "evidence: exact quote\n"
        "reason: why it may matter later\n\n"
        "CARRYOVER\n"
        "phrase: pronoun or carried-over phrase\n"
        "likely_subject: self|name|unclear\n"
        "evidence: exact quote\n"
        "reason: short reason\n\n"
        "DURABILITY\n"
        "text: what kinds of facts are durable in this note\n"
        "reason: short reason\n\n"
        "Keep reasons short. The journal author/self is not a named person entity. "
        "Only add aliases when this note itself supports the mapping. Do not merge names just because they share a first letter.\n\n"
        f"Date: {date}\n--- NOTE ---\n{note_text}\n--- END NOTE ---\n"
    )


def parse_block_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip().strip('"')
    return fields


def parse_context_blocks(text: str) -> str:
    context = {
        "people": [],
        "pets": [],
        "aliases": [],
        "active_topics": [],
        "subject_carryovers": [],
        "durability_notes": [],
        "warnings": [],
    }
    parts = re.split(r"(?im)^\s*(PERSON|PET|ALIAS|TOPIC|CARRYOVER|DURABILITY)\s*$", text)
    if len(parts) < 3:
        context["warnings"].append({"warning": "no context blocks found", "text": text[:300]})
        return json.dumps(context, indent=2)
    for idx in range(1, len(parts), 2):
        kind = parts[idx].strip().upper()
        fields = parse_block_fields(parts[idx + 1])
        if kind == "PERSON" and fields.get("name"):
            context["people"].append({
                "name": fields.get("name", ""),
                "evidence": fields.get("evidence", ""),
                "reason": fields.get("reason", ""),
            })
        elif kind == "PET" and fields.get("name"):
            context["pets"].append({
                "name": fields.get("name", ""),
                "evidence": fields.get("evidence", ""),
                "reason": fields.get("reason", ""),
            })
        elif kind == "ALIAS" and fields.get("alias"):
            evidence = fields.get("evidence", "")
            context["aliases"].append({
                "alias": fields.get("alias", ""),
                "canonical_name": fields.get("canonical_name", ""),
                "evidence": [evidence] if evidence else [],
                "confidence": fields.get("confidence", "low"),
                "reason": fields.get("reason", ""),
            })
        elif kind == "TOPIC" and fields.get("topic"):
            context["active_topics"].append({
                "topic": fields.get("topic", ""),
                "evidence": fields.get("evidence", ""),
                "reason": fields.get("reason", ""),
            })
        elif kind == "CARRYOVER" and fields.get("phrase"):
            context["subject_carryovers"].append({
                "phrase": fields.get("phrase", ""),
                "likely_subject": fields.get("likely_subject", "unclear"),
                "evidence": fields.get("evidence", ""),
                "reason": fields.get("reason", ""),
            })
        elif kind == "DURABILITY" and fields.get("text"):
            context["durability_notes"].append({
                "text": fields.get("text", ""),
                "reason": fields.get("reason", ""),
            })
    return json.dumps(context, indent=2)


def raw_prompt(date: str, context: str, label: str, text: str, idx: int, total: int,
               max_items: int) -> str:
    return (
        "Stage 2: extract raw durable memory candidates. Do not choose pillars or file paths. "
        "Output JSON only, no markdown.\n\n"
        "Schema:\n"
        "{\n"
        "  \"raw_items\": [{\"raw_id\":\"r1\",\"fact\":\"short literal fact\","
        "\"subject\":\"self|name|topic|unclear\",\"evidence_span\":\"exact quote\","
        "\"subject_evidence\":\"exact quote\",\"confidence\":\"high|medium|low\","
        "\"why_durable\":\"short reason\",\"uncertainty\":\"\"}],\n"
        "  \"entities\": [{\"name\":\"...\",\"type\":\"person|pet\",\"evidence\":\"exact quote\"}],\n"
        "  \"drops\": [{\"text\":\"...\",\"reason\":\"low-signal|not durable|duplicate|too transient\"}]\n"
        "}\n\n"
        f"Return at most {max_items} raw_items. Prefer durable facts, recurring topics, decisions, relationships, "
        "measurements, projects, preferences, and strong self-observations. Do not list every schedule line. "
        "Use the whole-note context to resolve carryover, but mark uncertainty when weak.\n\n"
        f"Date: {date}\nUnit: {label} ({idx}/{total})\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"--- TEXT ---\n{text}\n--- END TEXT ---\n"
    )


def raw_blocks_prompt(date: str, context: str, label: str, text: str, idx: int, total: int,
                      max_items: int) -> str:
    return (
        "Stage 2: extract raw durable memory candidates. Do not choose pillars or file paths. "
        "Use plain text blocks, not JSON or markdown.\n\n"
        "Write one block per durable fact in exactly this shape:\n"
        "FACT r1\n"
        "subject: self|name|topic|unclear\n"
        "fact: short literal fact\n"
        "evidence: exact quote from the source\n"
        "subject_evidence: exact quote showing the subject, or same as evidence\n"
        "confidence: high|medium|low\n"
        "why_durable: short reason\n"
        "uncertainty: empty or short note\n\n"
        "Optional entity lines can appear after a fact:\n"
        "entity: person|pet | Name | exact quote\n\n"
        f"Return at most {max_items} FACT blocks. Prefer durable facts, recurring topics, decisions, relationships, "
        "measurements, projects, preferences, and strong self-observations. Do not list every schedule line. "
        "Use the whole-note context to resolve carryover, but mark uncertainty when weak.\n\n"
        f"Date: {date}\nUnit: {label} ({idx}/{total})\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"--- TEXT ---\n{text}\n--- END TEXT ---\n"
    )


def final_fact_blocks_prompt(date: str, context: str, label: str, text: str, idx: int, total: int,
                             max_items: int) -> str:
    return (
        "Stage 2: extract final durable memory facts. Do not choose pillars or file paths. "
        "Use plain text blocks, not JSON or markdown.\n\n"
        "Write one block per final fact in exactly this shape:\n"
        "FACT n1\n"
        "kind: person_fact|self_state|project_fact|idea|event|preference|relationship|task\n"
        "subject: self|name|topic|unclear\n"
        "fact: short final fact\n"
        "evidence: exact quote from the source\n"
        "subject_evidence: exact quote showing the subject, or same as evidence\n"
        "confidence: high|medium|low\n"
        "decision_reason: short reason for keeping this fact\n"
        "uncertainty: empty or short note\n\n"
        "Optional entity lines can appear after a fact:\n"
        "entity: person|pet | Name | exact quote\n\n"
        f"Return at most {max_items} FACT blocks. Merge duplicates yourself. Collapse ordinary schedules. "
        "Prefer durable facts, recurring topics, decisions, relationships, measurements, projects, preferences, "
        "and strong self-observations. Use the whole-note context to resolve carryover, but mark uncertainty when weak.\n\n"
        f"Date: {date}\nUnit: {label} ({idx}/{total})\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"--- TEXT ---\n{text}\n--- END TEXT ---\n"
    )


def parse_raw_blocks(text: str, segment: int) -> dict:
    parsed = {"raw_items": [], "entities": [], "drops": [], "warnings": []}
    blocks = re.split(r"(?im)^\s*FACT\s+([A-Za-z0-9_-]+)\s*$", text)
    if len(blocks) < 3:
        parsed["warnings"].append({"segment": segment, "warning": "no FACT blocks found", "text": text[:300]})
        return parsed
    for idx in range(1, len(blocks), 2):
        raw_id = blocks[idx].strip()
        body = blocks[idx + 1]
        fields: dict[str, str] = {}
        entities = []
        for line in body.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip().strip('"')
            if key == "entity":
                parts = [part.strip().strip('"') for part in value.split("|")]
                if len(parts) >= 3:
                    entities.append({"type": parts[0], "name": parts[1], "evidence": parts[2]})
                continue
            fields[key] = value
        fact = fields.get("fact", "")
        evidence = fields.get("evidence", "")
        if not fact:
            parsed["warnings"].append({"segment": segment, "raw_id": raw_id, "warning": "FACT block missing fact"})
            continue
        parsed["raw_items"].append({
            "raw_id": raw_id,
            "fact": fact,
            "subject": fields.get("subject", "unclear"),
            "evidence_span": evidence,
            "subject_evidence": fields.get("subject_evidence", evidence),
            "confidence": fields.get("confidence", "medium"),
            "why_durable": fields.get("why_durable", ""),
            "uncertainty": fields.get("uncertainty", ""),
            "segment": segment,
        })
        parsed["entities"].extend(entities)
    return parsed


def parse_final_fact_blocks(text: str, segment: int) -> dict:
    parsed = {"normalized_items": [], "entities": [], "dropped_raw_ids": [], "warnings": []}
    blocks = re.split(r"(?im)^\s*FACT\s+([A-Za-z0-9_-]+)\s*$", text)
    if len(blocks) < 3:
        parsed["warnings"].append({"segment": segment, "warning": "no FACT blocks found", "text": text[:300]})
        return parsed
    for idx in range(1, len(blocks), 2):
        item_id = blocks[idx].strip()
        body = blocks[idx + 1]
        fields: dict[str, str] = {}
        entities = []
        for line in body.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip().strip('"')
            if key == "entity":
                parts = [part.strip().strip('"') for part in value.split("|")]
                if len(parts) >= 3:
                    entities.append({"type": parts[0], "name": parts[1], "evidence": parts[2]})
                continue
            fields[key] = value
        fact = fields.get("fact", "")
        evidence = fields.get("evidence", "")
        if not fact:
            parsed["warnings"].append({"segment": segment, "item_id": item_id, "warning": "FACT block missing fact"})
            continue
        parsed["normalized_items"].append({
            "item_id": item_id,
            "raw_ids": [],
            "keep": True,
            "memory_kind": fields.get("kind", fields.get("memory_kind", "self_state")),
            "fact": fact,
            "subject": fields.get("subject", "unclear"),
            "evidence_span": evidence,
            "subject_evidence": fields.get("subject_evidence", evidence),
            "confidence": fields.get("confidence", "medium"),
            "decision_reason": fields.get("decision_reason", ""),
            "uncertainty": fields.get("uncertainty", ""),
            "segment": segment,
        })
        parsed["entities"].extend(entities)
    return parsed


def merge_raw(raw_outputs: list[str]) -> str:
    merged = {"raw_items": [], "entities": [], "drops": [], "warnings": []}
    seen_entities: set[tuple[str, str]] = set()
    raw_counter = 0
    for seg_no, raw in enumerate(raw_outputs, 1):
        data = parse_json_object(raw)
        if not data:
            merged["warnings"].append({"segment": seg_no, "warning": "raw stage did not parse", "text": raw[:300]})
            continue
        for item in data.get("raw_items", []) if isinstance(data.get("raw_items"), list) else []:
            if not isinstance(item, dict):
                continue
            raw_counter += 1
            item["raw_id"] = f"r{raw_counter}"
            item["segment"] = seg_no
            merged["raw_items"].append(item)
        for ent in data.get("entities", []) if isinstance(data.get("entities"), list) else []:
            if not isinstance(ent, dict):
                continue
            key = (str(ent.get("name", "")), str(ent.get("type", "")))
            if key not in seen_entities:
                seen_entities.add(key)
                merged["entities"].append(ent)
        for drop in data.get("drops", []) if isinstance(data.get("drops"), list) else []:
            if isinstance(drop, dict):
                drop["segment"] = seg_no
                merged["drops"].append(drop)
    return json.dumps(merged, indent=2)


def merge_final_fact_blocks(raw_outputs: list[str]) -> str:
    merged = {"entities": [], "normalized_items": [], "dropped_raw_ids": [], "warnings": []}
    seen_entities: set[tuple[str, str]] = set()
    seen_facts: set[str] = set()
    item_counter = 0
    raw_counter = 0
    for seg_no, raw in enumerate(raw_outputs, 1):
        data = parse_final_fact_blocks(raw, seg_no)
        merged["warnings"].extend(data.get("warnings", []))
        for item in data.get("normalized_items", []):
            fact_key = re.sub(r"\s+", " ", str(item.get("fact", "")).strip().lower())
            if not fact_key or fact_key in seen_facts:
                continue
            seen_facts.add(fact_key)
            item_counter += 1
            raw_counter += 1
            item["item_id"] = f"n{item_counter}"
            item["raw_ids"] = [f"r{raw_counter}"]
            merged["normalized_items"].append(item)
        for ent in data.get("entities", []):
            key = (str(ent.get("name", "")), str(ent.get("type", "")))
            if key not in seen_entities:
                seen_entities.add(key)
                merged["entities"].append(ent)
    return json.dumps(merged, indent=2)


def raw_from_normalized(normalized: str) -> str:
    data = parse_json_object(normalized)
    raw_items = []
    for item in data.get("normalized_items", []) if isinstance(data.get("normalized_items"), list) else []:
        if not isinstance(item, dict):
            continue
        raw_id = (item.get("raw_ids") or [""])[0] if isinstance(item.get("raw_ids"), list) else ""
        raw_items.append({
            "raw_id": raw_id,
            "fact": item.get("fact", ""),
            "subject": item.get("subject", ""),
            "evidence_span": item.get("evidence_span", ""),
            "subject_evidence": item.get("subject_evidence", ""),
            "confidence": item.get("confidence", ""),
            "why_durable": item.get("decision_reason", ""),
            "uncertainty": item.get("uncertainty", ""),
            "segment": item.get("segment", ""),
        })
    return json.dumps({
        "raw_items": raw_items,
        "entities": data.get("entities", []),
        "drops": [],
        "warnings": data.get("warnings", []),
    }, indent=2)


def write_generated_stage(out_dir: Path, stage: str, output: str) -> None:
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "output.json").write_text(output.rstrip() + "\n", encoding="utf-8")
    meta = {"seconds": 0, "tokens": 0, "parsed": bool(parse_json_object(output)), "generated_by_harness": True}
    (stage_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def merge_raw_blocks(raw_outputs: list[str]) -> str:
    merged = {"raw_items": [], "entities": [], "drops": [], "warnings": []}
    seen_entities: set[tuple[str, str]] = set()
    raw_counter = 0
    for seg_no, raw in enumerate(raw_outputs, 1):
        data = parse_raw_blocks(raw, seg_no)
        merged["warnings"].extend(data.get("warnings", []))
        for item in data.get("raw_items", []):
            raw_counter += 1
            item["raw_id"] = f"r{raw_counter}"
            merged["raw_items"].append(item)
        for ent in data.get("entities", []):
            key = (str(ent.get("name", "")), str(ent.get("type", "")))
            if key not in seen_entities:
                seen_entities.add(key)
                merged["entities"].append(ent)
    return json.dumps(merged, indent=2)


def normalize_prompt(date: str, note_text: str, context: str, raw: str, max_items: int) -> str:
    return (
        "Stage 3: normalize and deduplicate raw candidates. Do not choose wiki paths yet. "
        "Output JSON only, no markdown.\n\n"
        "Schema:\n"
        "{\n"
        "  \"entities\": [{\"name\":\"...\",\"type\":\"person|pet\",\"evidence\":\"exact quote\"}],\n"
        "  \"normalized_items\": [{\"item_id\":\"n1\",\"raw_ids\":[\"r1\"],\"keep\":true,"
        "\"memory_kind\":\"person_fact|self_state|project_fact|idea|event|preference|relationship|task\","
        "\"fact\":\"short final fact\",\"subject\":\"self|name|topic|unclear\","
        "\"evidence_span\":\"exact quote\",\"subject_evidence\":\"exact quote\","
        "\"confidence\":\"high|medium|low\",\"decision_reason\":\"short reason\",\"uncertainty\":\"\"}],\n"
        "  \"dropped_raw_ids\": [{\"raw_id\":\"r1\",\"reason\":\"duplicate|not durable|too transient|unsupported\"}]\n"
        "}\n\n"
        f"Return at most {max_items} kept normalized_items. Merge duplicates. Collapse ordinary daily schedules. "
        "Keep uncertain but important facts with lower confidence instead of guessing.\n\n"
        f"Date: {date}\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\nRAW CANDIDATES:\n{raw}\n\n"
        f"--- SOURCE NOTE ---\n{note_text}\n--- END SOURCE NOTE ---\n"
    )


def placement_prompt(date: str, note_text: str, context: str, normalized: str) -> str:
    return (
        "Stage 4: choose JournalOS placement for normalized memory items. Output JSON only, no markdown. "
        "Do not repeat the fact text, evidence, citation, confidence, or subject; those are already in normalized items.\n\n"
        "Schema:\n"
        "{\n"
        "  \"entities\": [{\"name\":\"...\",\"type\":\"person|pet\",\"canonical_path\":\"Wiki/social-connections/profiles/<Name>.md\"}],\n"
        "  \"placement_decisions\": [{\"item_id\":\"n1\","
        "\"primary_home\":{\"pillar\":\"pillar-slug\",\"page_title\":\"durable object title\"}|null,"
        "\"primary_profile_name\":\"person or pet name|null\","
        "\"section\":\"Facts\",\"primary_reason\":\"short retrieval reason\","
        "\"profile_side_effects\":[{\"name\":\"...\",\"fact\":\"short cited profile bullet without citation\",\"reason\":\"short\"}],"
        "\"no_primary_placement_reason\":\"short or empty\","
        "\"no_profile_update_reason\":\"short or empty\"}],\n"
        "  \"uncertain\": [{\"item_id\":\"n1\",\"reason\":\"short\"}]\n"
        "}\n\n"
        "Rules:\n"
        "- Return exactly one placement_decision for every normalized item_id. Do not skip items.\n"
        "- Use primary_home for the best retrieval home of the main memory. Give a concrete page_title, not only a pillar.\n"
        "- Use primary_profile_name only when the main memory is truly about that named person or pet.\n"
        "- Every durable item should have a primary_home or primary_profile_name. profile_side_effects do not replace the primary retrieval home.\n"
        "- Set both primary_home and primary_profile_name to null only when the item is relationship-only, too low-signal, or has no useful durable retrieval page.\n"
        "- If both primary fields are null, fill no_primary_placement_reason in plain English.\n"
        "- profile_side_effects are lightweight profile updates for named people/pets mentioned by the item.\n"
        "- The journal author/self is not a profile. Never use self as primary_profile_name or a profile_side_effect name.\n"
        "- If the item is about a named person or pet, primary_profile_name should usually be that name.\n"
        "- If the item is a self task/project that mentions a person, primary_home should be the task/project page and profile_side_effects should mention the person.\n"
        "- Action tasks involving a person, like review/text/call/email/message, are primarily the author's task. Use primary_home for the task and a profile_side_effect for the person.\n"
        "- If the item is a relationship between people, do not create a couple page. Use profile_side_effects for each person and leave the primary fields null.\n"
        "- Never create couple/relationship page titles like Mukul and Rash, relationship, marriage, or couple.\n"
        "- If no profile update is useful for a named person, fill no_profile_update_reason.\n\n"
        f"Date: {date}\nDIRECTORIES:\n{pillar_guide()}\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"NORMALIZED ITEMS:\n{normalized}\n\n--- SOURCE NOTE ---\n{note_text}\n--- END SOURCE NOTE ---\n"
    )


def source_neighborhood(note_text: str, evidence: str) -> str:
    evidence = str(evidence or "").strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", note_text) if p.strip()]
    if evidence:
        for idx, para in enumerate(paragraphs):
            if evidence in para:
                start = max(0, idx - 1)
                end = min(len(paragraphs), idx + 2)
                return "\n\n".join(paragraphs[start:end])
    return "\n\n".join(paragraphs[:3])[:2400]


def neighbor_facts(normalized: str, item_id: str) -> str:
    data = parse_json_object(normalized)
    rows = []
    for item in data.get("normalized_items", []) if isinstance(data.get("normalized_items"), list) else []:
        if not isinstance(item, dict) or str(item.get("item_id")) == str(item_id):
            continue
        rows.append({
            "item_id": item.get("item_id", ""),
            "memory_kind": item.get("memory_kind", ""),
            "subject": item.get("subject", ""),
            "fact": item.get("fact", ""),
        })
    return json.dumps(rows, indent=2)


def batched(items: list[dict], size: int) -> list[list[dict]]:
    size = max(1, size)
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def batch_placement_prompt(date: str, note_text: str, context: str, normalized: str, items: list[dict]) -> str:
    item_ids = [str(item.get("item_id") or "") for item in items if isinstance(item, dict)]
    target_items = [item for item in items if isinstance(item, dict)]
    return (
        "Stage 4: choose JournalOS placement for this small batch of normalized memory items. "
        "Output JSON only, no markdown. Do not place neighbor facts outside this batch.\n\n"
        "Schema:\n"
        "{\n"
        "  \"entities\": [{\"name\":\"...\",\"type\":\"person|pet\",\"canonical_path\":\"Wiki/social-connections/profiles/<Name>.md\"}],\n"
        "  \"placement_decisions\": [{\"item_id\":\"n1\","
        "\"primary_home\":{\"pillar\":\"pillar-slug\",\"page_title\":\"durable object title\"}|null,"
        "\"primary_profile_name\":\"person or pet name|null\","
        "\"section\":\"Facts\",\"primary_reason\":\"short retrieval reason\","
        "\"profile_side_effects\":[{\"name\":\"...\",\"fact\":\"short cited profile bullet without citation\",\"reason\":\"short\"}],"
        "\"no_primary_placement_reason\":\"short or empty\","
        "\"no_profile_update_reason\":\"short or empty\"}],\n"
        "  \"uncertain\": [{\"item_id\":\"n1\",\"reason\":\"short\"}]\n"
        "}\n\n"
        "Rules:\n"
        f"- Return exactly one placement_decision for each of these item_ids: {', '.join(item_ids)}.\n"
        "- Use primary_home for the best retrieval home of the main memory. Give a concrete page_title, not only a pillar.\n"
        "- Use primary_profile_name only when the main memory is truly about that named person or pet.\n"
        "- Every durable item should have a primary_home or primary_profile_name. profile_side_effects do not replace the primary retrieval home.\n"
        "- Leave both primary fields null only when the item is relationship-only, too low-signal, or has no useful durable retrieval page.\n"
        "- If both primary fields are null, fill no_primary_placement_reason in plain English.\n"
        "- The journal author/self is not a profile. Never use self as primary_profile_name or a profile_side_effect name.\n"
        "- Action tasks involving a person, like review/text/call/email/message, are primarily the author's task. Use primary_home for the task and a profile_side_effect for the person.\n"
        "- If the item is a relationship between people, do not create a couple page. Use profile_side_effects for each person and leave the primary fields null only when there is no better retrieval home.\n"
        "- Never create couple/relationship page titles like Mukul and Rash, relationship, marriage, or couple.\n\n"
        f"Date: {date}\nDIRECTORIES:\n{pillar_guide()}\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"TARGET ITEMS:\n{json.dumps(target_items, indent=2)}\n\n"
        f"NEIGHBOR FACTS FOR CONTEXT ONLY:\n{neighbor_facts(normalized, '')}\n\n"
        f"--- SOURCE NOTE ---\n{note_text}\n--- END SOURCE NOTE ---\n"
    )


def single_placement_prompt(date: str, note_text: str, context: str, normalized: str, item: dict) -> str:
    item_id = str(item.get("item_id") or "")
    return (
        "Stage 4: choose JournalOS placement for exactly one normalized memory item. "
        "Output JSON only, no markdown. Do not place neighbor facts.\n\n"
        "Schema:\n"
        "{\n"
        "  \"entities\": [{\"name\":\"...\",\"type\":\"person|pet\",\"canonical_path\":\"Wiki/social-connections/profiles/<Name>.md\"}],\n"
        "  \"placement_decisions\": [{\"item_id\":\"" + item_id + "\","
        "\"primary_home\":{\"pillar\":\"pillar-slug\",\"page_title\":\"durable object title\"}|null,"
        "\"primary_profile_name\":\"person or pet name|null\","
        "\"section\":\"Facts\",\"primary_reason\":\"short retrieval reason\","
        "\"profile_side_effects\":[{\"name\":\"...\",\"fact\":\"short cited profile bullet without citation\",\"reason\":\"short\"}],"
        "\"no_primary_placement_reason\":\"short or empty\","
        "\"no_profile_update_reason\":\"short or empty\"}],\n"
        "  \"uncertain\": [{\"item_id\":\"" + item_id + "\",\"reason\":\"short\"}]\n"
        "}\n\n"
        "Rules:\n"
        f"- Return exactly one placement_decision for item_id {item_id}.\n"
        "- Use primary_home for the best retrieval home of the main memory. Give a concrete page_title, not only a pillar.\n"
        "- Use primary_profile_name only when the main memory is truly about that named person or pet.\n"
        "- Every durable item should have a primary_home or primary_profile_name. profile_side_effects do not replace the primary retrieval home.\n"
        "- Leave both primary fields null only when the item is relationship-only, too low-signal, or has no useful durable retrieval page.\n"
        "- If both primary fields are null, fill no_primary_placement_reason in plain English.\n"
        "- The journal author/self is not a profile. Never use self as primary_profile_name or a profile_side_effect name.\n"
        "- Action tasks involving a person, like review/text/call/email/message, are primarily the author's task. Use primary_home for the task and a profile_side_effect for the person.\n"
        "- If the item is a relationship between people, do not create a couple page. Use profile_side_effects for each person and leave the primary fields null only when there is no better retrieval home.\n"
        "- Never create couple/relationship page titles like Mukul and Rash, relationship, marriage, or couple.\n\n"
        f"Date: {date}\nDIRECTORIES:\n{pillar_guide()}\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"TARGET ITEM:\n{json.dumps(item, indent=2)}\n\n"
        f"NEIGHBOR FACTS FOR CONTEXT ONLY:\n{neighbor_facts(normalized, item_id)}\n\n"
        f"SOURCE NEIGHBORHOOD:\n{source_neighborhood(note_text, item.get('evidence_span', ''))}\n"
    )


def primary_omission_retry_prompt(date: str, note_text: str, context: str, normalized: str,
                                  item: dict, omission_reason: str) -> str:
    item_id = str(item.get("item_id") or "")
    return (
        "Stage 4 retry: choose the primary JournalOS placement for this one durable fact. "
        "Output JSON only, no markdown.\n\n"
        "You previously left the primary page empty and gave this reason:\n"
        f"{omission_reason}\n\n"
        "That reason says the fact belongs in a career, research, project, or knowledge context. "
        "Choose the best primary_home now. Do not omit the primary page for this retry.\n\n"
        "Schema:\n"
        "{\n"
        "  \"entities\": [{\"name\":\"...\",\"type\":\"person|pet\",\"canonical_path\":\"Wiki/social-connections/profiles/<Name>.md\"}],\n"
        "  \"placement_decisions\": [{\"item_id\":\"" + item_id + "\","
        "\"primary_home\":{\"pillar\":\"pillar-slug\",\"page_title\":\"durable object title\"},"
        "\"primary_profile_name\":\"person or pet name|null\","
        "\"section\":\"Facts\",\"primary_reason\":\"short retrieval reason\","
        "\"profile_side_effects\":[{\"name\":\"...\",\"fact\":\"short cited profile bullet without citation\",\"reason\":\"short\"}],"
        "\"no_primary_placement_reason\":\"\","
        "\"no_profile_update_reason\":\"short or empty\"}],\n"
        "  \"uncertain\": []\n"
        "}\n\n"
        "Rules:\n"
        f"- Return exactly one placement_decision for item_id {item_id}.\n"
        "- Give a concrete primary_home page_title, not only a pillar.\n"
        "- The journal author/self is not a profile. Never use self as primary_profile_name.\n"
        "- Never create couple/relationship page titles like Mukul and Rash, relationship, marriage, or couple.\n\n"
        f"Date: {date}\nDIRECTORIES:\n{pillar_guide()}\n\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\n"
        f"TARGET ITEM:\n{json.dumps(item, indent=2)}\n\n"
        f"SOURCE NEIGHBORHOOD:\n{source_neighborhood(note_text, item.get('evidence_span', ''))}\n"
    )


def should_retry_primary_omission(omission: dict) -> bool:
    reason = str(omission.get("reason") or "").lower()
    fact = str(omission.get("fact") or "").lower()
    text = f"{reason} {fact}"
    retry_terms = (
        "career", "research", "project", "context", "knowledge", "training",
        "prediction", "model", "work", "direction", "preference",
    )
    allowed_omissions = (
        "too low-signal", "too_low_signal", "duplicate", "unsupported",
        "unclear", "relationship-only", "relationship_only",
    )
    return any(term in text for term in retry_terms) and not any(term in reason for term in allowed_omissions)


def should_agentic_repair_omission(omission: dict) -> bool:
    reason = str(omission.get("reason") or "").strip().lower()
    if not reason:
        return True
    allowed_omissions = (
        "too low-signal", "too_low_signal", "duplicate", "unsupported",
        "unclear", "relationship-only", "relationship_only", "transient",
    )
    return should_retry_primary_omission(omission) and not any(term in reason for term in allowed_omissions)


def replace_placement_decisions(original: str, retry_outputs: list[str]) -> str:
    data = parse_json_object(original)
    if not data:
        return original
    retry_data = [parse_json_object(output) for output in retry_outputs]
    retry_decisions = []
    retry_ids = set()
    for output in retry_data:
        decisions = output.get("placement_decisions", output.get("placements", [])) if output else []
        for decision in decisions if isinstance(decisions, list) else []:
            if isinstance(decision, dict) and decision.get("item_id"):
                retry_ids.add(str(decision.get("item_id")))
                retry_decisions.append(decision)
    if not retry_ids:
        return original
    decisions = data.get("placement_decisions", data.get("placements", []))
    kept = [
        decision for decision in decisions
        if isinstance(decision, dict) and str(decision.get("item_id")) not in retry_ids
    ] if isinstance(decisions, list) else []
    data["placement_decisions"] = kept + retry_decisions
    data["uncertain"] = data.get("uncertain", [])
    return json.dumps(data, indent=2)


def tokenize(text: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "through", "need", "needs",
        "ways", "make", "money", "like", "from", "into", "onto", "self", "fact",
        "have", "has", "was", "were", "are", "can", "could", "would", "should",
    }
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2 and tok not in stop]


def safe_wiki_path(vault: Path, rel: str):
    rel = str(rel or "").strip()
    if rel.startswith("/"):
        return None
    path = (vault / rel).resolve()
    if not str(path).startswith(str(vault.resolve())):
        return None
    return path


def placement_list_pillars() -> str:
    return pillar_guide()


def placement_search_pages(vault: Path, query: str, limit: int = 8) -> str:
    wiki = vault / "Wiki"
    if not wiki.exists():
        return "(no Wiki directory)"
    q = set(tokenize(query))
    results = []
    for path in wiki.rglob("*.md"):
        if path.name.startswith("."):
            continue
        rel = path.relative_to(vault)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        headings = " ".join(re.findall(r"^#+\s+(.+)$", text, flags=re.MULTILINE))
        hay = set(tokenize(f"{rel} {path.stem.replace('-', ' ')} {headings} {text[:1000]}"))
        score = len(q & hay)
        if score:
            snippet = " ".join(text.split())[:220]
            results.append((score, str(rel), snippet))
    results.sort(key=lambda row: (-row[0], row[1]))
    if not results:
        return "(no matches)"
    return "\n".join(f"{score} | {rel} | {snippet}" for score, rel, snippet in results[:limit])


def placement_read_page_summary(vault: Path, rel: str) -> str:
    path = safe_wiki_path(vault, rel)
    if not path or not path.exists() or not path.is_file():
        return "(no such page)"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return f"ERROR: {exc}"
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("- ") or line.startswith("* "):
            lines.append(line)
        if len("\n".join(lines)) > 1800:
            break
    return "\n".join(lines)[:2000] or text[:1000]


def parse_agentic_final(text: str, item_id: str) -> str:
    if not text.strip().upper().startswith("FINAL"):
        return ""
    fields = parse_block_fields(text)
    pillar = fields.get("primary_pillar", "")
    page = fields.get("primary_page", "")
    if not pillar or not page:
        return ""
    page_slug = slug(page)
    if page_slug in ("index", "index-md", slug(pillar)):
        return ""
    decision = {
        "item_id": item_id,
        "primary_home": {"pillar": pillar, "page_title": page},
        "primary_profile_name": fields.get("primary_profile") or None,
        "section": "Facts",
        "primary_reason": fields.get("reason", ""),
        "profile_side_effects": [],
        "no_primary_placement_reason": "",
        "no_profile_update_reason": fields.get("no_profile_update_reason", ""),
    }
    return json.dumps({"entities": [], "placement_decisions": [decision], "uncertain": []}, indent=2)


def agentic_placement_repair(out_dir: Path, stage: str, base_url: str, model: str,
                             vault: Path, date: str, item: dict, reason: str,
                             max_tokens: int) -> str:
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    item_id = str(item.get("item_id") or "")
    system = (
        "You are repairing placement for one extracted journal fact. "
        "Use read-only tools to find the best wiki page. "
        "Return exactly one command per turn, no markdown.\n\n"
        "Commands:\n"
        "LIST_PILLARS\n"
        "SEARCH <query>\n"
        "READ <Wiki/path.md>\n"
        "FINAL\n"
        "item_id: <id>\n"
        "primary_pillar: <pillar-slug>\n"
        "primary_page: <page title>\n"
        "primary_profile:\n"
        "existing_or_new: existing|new\n"
        "reason: <short reason>\n\n"
        "Rules:\n"
        "- Do not finalize to Index.md, index-md, or a broad pillar landing page.\n"
        "- A final page must be a specific topic, project, person/pet profile, habit, or problem page.\n"
        "- If search only finds index/landing pages, choose existing_or_new: new and create a specific page title.\n"
        "Use SEARCH/READ until you know the best page. If no existing page fits, choose a good new page."
    )
    user = (
        f"Date: {date}\nPlacement failed or omitted the primary page.\n"
        f"Failure reason: {reason or '(empty)'}\n\n"
        f"Fact item:\n{json.dumps(item, indent=2)}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    transcript = []
    final_text = ""
    for _ in range(8):
        content, _tokens = A.chat(base_url, model, messages, max_tokens=500, reasoning_budget=0, timeout=180)
        command = content.strip()
        transcript.append({"role": "model", "text": command})
        upper = command.upper()
        if upper.startswith("FINAL"):
            final_text = command
            break
        if upper.startswith("LIST_PILLARS"):
            observation = placement_list_pillars()
        elif upper.startswith("SEARCH"):
            query = command.split(" ", 1)[1] if " " in command else ""
            observation = placement_search_pages(vault, query)
        elif upper.startswith("READ"):
            rel = command.split(" ", 1)[1].strip() if " " in command else ""
            observation = placement_read_page_summary(vault, rel)
        else:
            observation = "ERROR: command must be LIST_PILLARS, SEARCH <query>, READ <Wiki/path.md>, or FINAL block."
        transcript.append({"role": "tool", "text": observation})
        messages.append({"role": "assistant", "content": command})
        messages.append({"role": "user", "content": f"OBSERVATION:\n{observation[:2400]}"})
    (stage_dir / "transcript.json").write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    output = parse_agentic_final(final_text, item_id)
    (stage_dir / "output.json").write_text((output or "{}").rstrip() + "\n", encoding="utf-8")
    (stage_dir / "meta.json").write_text(
        json.dumps({"parsed": bool(output), "turns": len([t for t in transcript if t["role"] == "model"])}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def combine_placement_decisions(outputs: list[str]) -> str:
    combined = {"entities": [], "placement_decisions": [], "uncertain": []}
    seen_entities: set[tuple[str, str]] = set()
    for output in outputs:
        data = parse_json_object(output)
        if not data:
            continue
        for ent in data.get("entities", []) if isinstance(data.get("entities"), list) else []:
            if not isinstance(ent, dict):
                continue
            key = (str(ent.get("name", "")), str(ent.get("type", "")))
            if key not in seen_entities:
                seen_entities.add(key)
                combined["entities"].append(ent)
        decisions = data.get("placement_decisions", data.get("placements", []))
        for decision in decisions if isinstance(decisions, list) else []:
            if isinstance(decision, dict):
                combined["placement_decisions"].append(decision)
        for uncertain in data.get("uncertain", []) if isinstance(data.get("uncertain"), list) else []:
            if isinstance(uncertain, dict):
                combined["uncertain"].append(uncertain)
    return json.dumps(combined, indent=2)


def merge_placement_decisions(date: str, normalized: str, placement_decisions: str, context: str = "") -> str:
    norm_data = parse_json_object(normalized)
    place_data = parse_json_object(placement_decisions)
    if not norm_data or not place_data:
        return placement_decisions
    aliases = alias_map(context)
    items_by_id = {
        str(item.get("item_id")): item
        for item in norm_data.get("normalized_items", [])
        if isinstance(item, dict) and item.get("item_id")
    }
    merged = {
        "entities": place_data.get("entities", norm_data.get("entities", [])),
        "primary_placements": [],
        "profile_side_effects": [],
        "primary_omissions": [],
        "placements": [],
        "uncertain": place_data.get("uncertain", []),
    }
    decisions = place_data.get("placement_decisions", place_data.get("placements", []))
    if not isinstance(decisions, list):
        return placement_decisions
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        item = items_by_id.get(str(decision.get("item_id")))
        if not item:
            merged["uncertain"].append({
                "item_id": decision.get("item_id", ""),
                "reason": "placement decision did not match a normalized item",
            })
            continue
        primary_home = decision.get("primary_home")
        primary_home_target = home_path(primary_home)
        primary_profile_name = canonical_name(str(decision.get("primary_profile_name") or "").strip(), aliases)
        if primary_profile_name.lower() in ("null", "none", "self"):
            primary_profile_name = ""
        profile_title = (
            primary_profile_name
            and item.get("memory_kind") in ("person_fact", "pet_fact")
            and "profile" in home_title(primary_home).lower()
        )
        if primary_profile_name and profile_title:
            primary_target = profile_path(primary_profile_name)
        elif primary_home_target:
            primary_target = primary_home_target
        elif action_task_path(item, primary_profile_name):
            primary_target = action_task_path(item, primary_profile_name)
        elif primary_profile_name:
            primary_target = profile_path(primary_profile_name)
        else:
            primary_target = str(decision.get("primary_target", decision.get("target_hint", "")) or "")
        side_effects = decision.get("profile_side_effects", [])
        merged_item = {
            "item_id": item.get("item_id", ""),
            "memory_kind": item.get("memory_kind", ""),
            "subject": item.get("subject", ""),
            "primary_target": primary_target,
            "target_hint": primary_target,
            "section": decision.get("section", "Facts") or "Facts",
            "fact": item.get("fact", ""),
            "citation": f"[[{date}]]",
            "evidence_span": item.get("evidence_span", ""),
            "subject_evidence": item.get("subject_evidence", ""),
            "confidence": item.get("confidence", ""),
            "primary_reason": decision.get("primary_reason", decision.get("placement_reason", "")),
            "placement_reason": decision.get("primary_reason", decision.get("placement_reason", "")),
            "no_primary_placement_reason": decision.get("no_primary_placement_reason", ""),
            "no_profile_update_reason": decision.get("no_profile_update_reason", ""),
        }
        if primary_target:
            merged["primary_placements"].append(merged_item)
            merged["placements"].append(merged_item)
        else:
            merged["primary_omissions"].append({
                "item_id": item.get("item_id", ""),
                "memory_kind": item.get("memory_kind", ""),
                "subject": item.get("subject", ""),
                "fact": item.get("fact", ""),
                "reason": decision.get("no_primary_placement_reason", ""),
                "profile_side_effect_count": len(side_effects) if isinstance(side_effects, list) else 0,
            })
        if isinstance(side_effects, list):
            for effect in side_effects:
                if not isinstance(effect, dict):
                    continue
                name = canonical_name(str(effect.get("name") or "").strip(), aliases)
                if not name:
                    merged["uncertain"].append({
                        "item_id": item.get("item_id", ""),
                        "reason": "ignored profile side effect with no person or pet name",
                    })
                    continue
                if name.lower() in ("self", "journal author", "author", "me", "i"):
                    merged["uncertain"].append({
                        "item_id": item.get("item_id", ""),
                        "reason": f"ignored invalid self profile side effect: {name}",
                    })
                    continue
                fact = str(effect.get("fact") or "").strip()
                if fact and "[[" not in fact:
                    fact = f"{fact.rstrip('.')}." if not fact.endswith(".") else fact
                    fact = f"{fact} ([[{date}]])"
                merged["profile_side_effects"].append({
                    "item_id": item.get("item_id", ""),
                    "name": name,
                    "path": profile_path(name),
                    "fact": fact,
                    "reason": effect.get("reason", ""),
                    "source_fact": item.get("fact", ""),
                })
    return json.dumps(merged, indent=2)


def review_prompt(date: str, note_text: str, context: str, raw: str, normalized: str, placements: str) -> str:
    return (
        "Stage 5: review this proposed ingest plan. You are a reviewer, not a writer. "
        "Output JSON only, no markdown.\n\n"
        "Schema:\n"
        "{\n"
        "  \"verdict\":\"ok|needs_revision\",\n"
        "  \"findings\": [{\"stage\":\"context|raw|normalize|placement\","
        "\"severity\":\"info|warn|error\",\"item_id\":\"n1|r1|\",\"finding\":\"short\","
        "\"review_reason\":\"short reason\",\"suggested_fix\":\"short\"}],\n"
        "  \"coverage_notes\": [{\"text\":\"durable source area that may be missed\",\"review_reason\":\"short\"}]\n"
        "}\n\n"
        "Check subject attribution, missing durable facts, over-extracted transient schedules, unsupported evidence, "
        "wrong relationship direction, weak named-person health/body facts, bad primary targets, missing profile side effects, "
        "profile side effects that should be primary targets, and relationship/couple profile pages.\n\n"
        f"Date: {date}\nWHOLE-NOTE CONTEXT:\n{context or '{}'}\n\nRAW:\n{raw}\n\n"
        f"NORMALIZED:\n{normalized}\n\nPLACEMENTS:\n{placements}\n\n--- SOURCE NOTE ---\n{note_text}\n--- END SOURCE NOTE ---\n"
    )


def validate_outputs(note_text: str, raw: str, normalized: str, placements: str, review: str) -> list[str]:
    warnings: list[str] = []
    for stage, text in (("raw", raw), ("normalize", normalized), ("placement", placements), ("review", review)):
        if not parse_json_object(text):
            warnings.append(f"{stage}: output did not parse as JSON")
    placement_data = parse_json_object(placements)
    norm_data = parse_json_object(normalized)
    normalized_ids = {
        str(item.get("item_id"))
        for item in norm_data.get("normalized_items", [])
        if isinstance(item, dict) and item.get("item_id")
    } if norm_data else set()
    placement_ids = {
        str(item.get("item_id"))
        for item in placement_data.get("placements", [])
        if isinstance(item, dict) and item.get("item_id")
    } if placement_data else set()
    side_effect_ids = {
        str(item.get("item_id"))
        for item in placement_data.get("profile_side_effects", [])
        if isinstance(item, dict) and item.get("item_id")
    } if placement_data else set()
    omission_ids = {
        str(item.get("item_id"))
        for item in placement_data.get("primary_omissions", [])
        if isinstance(item, dict) and item.get("item_id") and str(item.get("reason") or "").strip()
    } if placement_data else set()
    missing_ids = normalized_ids - placement_ids - side_effect_ids - omission_ids
    if missing_ids:
        warnings.append(f"placement: missing decisions for item_ids {sorted(missing_ids)}")
    missing_primary_ids = normalized_ids - placement_ids - omission_ids
    if missing_primary_ids:
        warnings.append(f"placement: missing primary placement or explicit reason for item_ids {sorted(missing_primary_ids)}")
    for idx, omission in enumerate(placement_data.get("primary_omissions", []) if isinstance(placement_data.get("primary_omissions"), list) else [], 1):
        if not isinstance(omission, dict):
            warnings.append(f"primary omission {idx}: not an object")
            continue
        if not str(omission.get("reason") or "").strip():
            warnings.append(f"primary omission {idx}: missing no_primary_placement_reason")
    for idx, item in enumerate(placement_data.get("placements", []) if isinstance(placement_data.get("placements"), list) else [], 1):
        if not isinstance(item, dict):
            warnings.append(f"placement item {idx}: not an object")
            continue
        for field in ("evidence_span", "subject_evidence"):
            val = str(item.get(field) or "")
            if val and val not in note_text:
                warnings.append(f"placement item {idx}: {field} is not an exact source quote")
        target = str(item.get("target_hint") or "")
        if not target.startswith("Wiki/") or not target.endswith(".md"):
            warnings.append(f"placement item {idx}: target_hint is not a wiki markdown path")
        elif len(Path(target).parts) == 2:
            warnings.append(f"placement item {idx}: target_hint is a bare pillar file, not a durable object page")
        stem = Path(target).stem.lower()
        if "/profiles/" in target and any(word in stem for word in ("and", "relationship", "couple", "marriage")):
            warnings.append(f"placement item {idx}: relationship/couple profile path is not allowed")
    for idx, effect in enumerate(placement_data.get("profile_side_effects", []) if isinstance(placement_data.get("profile_side_effects"), list) else [], 1):
        if not isinstance(effect, dict):
            warnings.append(f"profile side effect {idx}: not an object")
            continue
        path = str(effect.get("path") or "")
        if not path.startswith("Wiki/social-connections/profiles/") or not path.endswith(".md"):
            warnings.append(f"profile side effect {idx}: path is not a profile markdown path")
        stem = Path(path).stem.lower()
        if any(word in stem for word in ("and", "relationship", "couple", "marriage")):
            warnings.append(f"profile side effect {idx}: relationship/couple profile path is not allowed")
        fact = str(effect.get("fact") or "")
        if "[[" not in fact:
            warnings.append(f"profile side effect {idx}: fact is missing citation")
    return warnings


def run_note(note_path: Path, out_root: Path, model: str, base_url: str, mode: str,
             max_tokens: int, segment_chars: int, raw_max_items: int,
             normalize_max_items: int, placement_mode: str = "list",
             placement_batch_size: int = 4, raw_format: str = "json",
             context_format: str = "json", placement_vault: str = "",
             placement_repair_mode: str = "prompt") -> dict:
    date = note_path.stem
    note_text = note_path.read_text(encoding="utf-8")
    out_dir = out_root / date
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log(f"=== {date} ({mode}) ===")

    if context_format == "blocks":
        context_text = call_text_stage(
            out_dir, "01-context", base_url, model,
            context_blocks_prompt(date, note_text), max_tokens, 240
        )
        context = parse_context_blocks(context_text)
        (out_dir / "01-context" / "output.json").write_text(context.rstrip() + "\n", encoding="utf-8")
    else:
        context = call_stage(out_dir, "01-context", base_url, model, context_prompt(date, note_text), max_tokens, 240)

    segments = segment_text(note_text, segment_chars if mode == "chunk" else 0)
    raw_outputs = []
    for idx, segment in enumerate(segments, 1):
        if raw_format == "final-blocks":
            raw_outputs.append(call_text_stage(
                out_dir, f"02-facts-{idx:02d}", base_url, model,
                final_fact_blocks_prompt(date, context, "full note" if mode == "full" else f"chunk {idx}", segment, idx, len(segments), raw_max_items),
                max_tokens, 240
            ))
        elif raw_format == "blocks":
            raw_outputs.append(call_text_stage(
                out_dir, f"02-raw-{idx:02d}", base_url, model,
                raw_blocks_prompt(date, context, "full note" if mode == "full" else f"chunk {idx}", segment, idx, len(segments), raw_max_items),
                max_tokens, 240
            ))
        else:
            raw_outputs.append(call_stage(
                out_dir, f"02-raw-{idx:02d}", base_url, model,
                raw_prompt(date, context, "full note" if mode == "full" else f"chunk {idx}", segment, idx, len(segments), raw_max_items),
                max_tokens, 240
            ))
    if raw_format == "final-blocks":
        normalized = merge_final_fact_blocks(raw_outputs)
        raw = raw_from_normalized(normalized)
    else:
        raw = merge_raw_blocks(raw_outputs) if raw_format == "blocks" else merge_raw(raw_outputs)
    (out_dir / "02-raw-merged.json").write_text(raw + "\n", encoding="utf-8")

    if raw_format == "final-blocks":
        write_generated_stage(out_dir, "03-normalize", normalized)
    else:
        normalized = call_stage(
            out_dir, "03-normalize", base_url, model,
            normalize_prompt(date, note_text, context, raw, normalize_max_items),
            max_tokens, 300
        )
    if placement_mode in ("item", "batch", "agentic"):
        norm_data = parse_json_object(normalized)
        norm_items = [
            item for item in norm_data.get("normalized_items", [])
            if isinstance(item, dict)
        ] if norm_data else []
        placement_outputs = []
        if placement_mode == "agentic":
            vault = Path(placement_vault).expanduser().resolve() if placement_vault else out_root
            for item in norm_items:
                item_id = str(item.get("item_id") or "unknown")
                placement_outputs.append(agentic_placement_repair(
                    out_dir, f"04-placement-agentic-{item_id}", base_url, model,
                    vault, date, item, "normal agentic placement", max_tokens
                ))
        elif placement_mode == "item":
            for item in norm_items:
                item_id = str(item.get("item_id") or "unknown")
                placement_outputs.append(call_stage(
                    out_dir, f"04-placement-{item_id}", base_url, model,
                    single_placement_prompt(date, note_text, context, normalized, item),
                    max_tokens, 240
                ))
        else:
            for idx, item_batch in enumerate(batched(norm_items, placement_batch_size), 1):
                batch_output = call_stage(
                    out_dir, f"04-placement-batch-{idx:02d}", base_url, model,
                    batch_placement_prompt(date, note_text, context, normalized, item_batch),
                    max_tokens, 300
                )
                if parse_json_object(batch_output):
                    placement_outputs.append(batch_output)
                    continue
                for item in item_batch:
                    item_id = str(item.get("item_id") or "unknown")
                    placement_outputs.append(call_stage(
                        out_dir, f"04-placement-batch-{idx:02d}-fallback-{item_id}", base_url, model,
                        single_placement_prompt(date, note_text, context, normalized, item),
                        max_tokens, 240
                    ))
        placement_decisions = combine_placement_decisions(placement_outputs)
        (out_dir / "04-placement-output-combined.json").write_text(placement_decisions.rstrip() + "\n", encoding="utf-8")
    else:
        placement_decisions = call_stage(
            out_dir, "04-placement", base_url, model,
            placement_prompt(date, note_text, context, normalized),
            max_tokens, 300
        )
    placements = merge_placement_decisions(date, normalized, placement_decisions, context)
    placement_data = parse_json_object(placements)
    norm_data_for_retry = parse_json_object(normalized)
    items_by_id_for_retry = {
        str(item.get("item_id")): item
        for item in norm_data_for_retry.get("normalized_items", [])
        if isinstance(item, dict) and item.get("item_id")
    } if norm_data_for_retry else {}
    if placement_repair_mode != "none":
        placed_ids = {
            str(item.get("item_id"))
            for item in placement_data.get("placements", [])
            if isinstance(item, dict) and item.get("item_id")
        } if placement_data else set()
        valid_omission_ids = {
            str(item.get("item_id"))
            for item in placement_data.get("primary_omissions", [])
            if isinstance(item, dict) and item.get("item_id") and str(item.get("reason") or "").strip()
            and not should_agentic_repair_omission(item)
        } if placement_data else set()
        repair_items: list[tuple[dict, str]] = []
        for item_id, item in items_by_id_for_retry.items():
            if item_id not in placed_ids and item_id not in valid_omission_ids:
                repair_items.append((item, "missing or invalid primary placement"))
        for omission in placement_data.get("primary_omissions", []) if placement_data else []:
            if not isinstance(omission, dict) or not should_agentic_repair_omission(omission):
                continue
            item = items_by_id_for_retry.get(str(omission.get("item_id")))
            if item and str(item.get("item_id")) not in {str(existing.get("item_id")) for existing, _ in repair_items}:
                repair_items.append((item, str(omission.get("reason") or "")))

        retry_outputs = []
        vault = Path(placement_vault).expanduser().resolve() if placement_vault else out_root
        for item, reason in repair_items:
            item_id = str(item.get("item_id") or "unknown")
            if placement_repair_mode == "agentic":
                agentic_output = agentic_placement_repair(
                    out_dir, f"04-placement-agentic-repair-{item_id}", base_url, model,
                    vault, date, item, reason, max_tokens
                )
                if agentic_output and parse_json_object(agentic_output):
                    retry_outputs.append(agentic_output)
                else:
                    retry_outputs.append(call_stage(
                        out_dir, f"04-placement-agentic-direct-fallback-{item_id}", base_url, model,
                        primary_omission_retry_prompt(date, note_text, context, normalized, item, reason),
                        max_tokens, 240
                    ))
            elif should_retry_primary_omission({"item_id": item_id, "fact": item.get("fact", ""), "reason": reason}):
                retry_outputs.append(call_stage(
                    out_dir, f"04-placement-primary-retry-{item_id}", base_url, model,
                    primary_omission_retry_prompt(date, note_text, context, normalized, item, reason),
                    max_tokens, 240
                ))
        retry_outputs = [output for output in retry_outputs if output and parse_json_object(output)]
        if retry_outputs:
            placement_decisions = replace_placement_decisions(placement_decisions, retry_outputs)
            (out_dir / "04-placement-output-retried.json").write_text(placement_decisions.rstrip() + "\n", encoding="utf-8")
            placements = merge_placement_decisions(date, normalized, placement_decisions, context)
    (out_dir / "04-placement-merged.json").write_text(placements.rstrip() + "\n", encoding="utf-8")
    review = call_stage(
        out_dir, "05-review", base_url, model,
        review_prompt(date, note_text, context, raw, normalized, placements),
        max_tokens, 300
    )

    warnings = validate_outputs(note_text, raw, normalized, placements, review)
    raw_data = parse_json_object(raw)
    norm_data = parse_json_object(normalized)
    place_data = parse_json_object(placements)
    review_data = parse_json_object(review)
    result = {
        "date": date,
        "mode": mode,
        "raw_items": len(raw_data.get("raw_items", [])) if isinstance(raw_data.get("raw_items"), list) else 0,
        "normalized_items": len(norm_data.get("normalized_items", [])) if isinstance(norm_data.get("normalized_items"), list) else 0,
        "placements": len(place_data.get("placements", [])) if isinstance(place_data.get("placements"), list) else 0,
        "review_verdict": review_data.get("verdict", "unparsed") if review_data else "unparsed",
        "warnings": warnings,
        "seconds": elapsed(t0),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"   summary: raw={result['raw_items']} normalized={result['normalized_items']} placements={result['placements']} verdict={result['review_verdict']} warnings={len(warnings)} time={result['seconds']}s")
    return result


def journal_notes(paths: list[str]) -> list[Path]:
    notes: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            notes.extend(sorted(x for x in p.rglob("*.md") if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", x.name)))
        elif p.is_file():
            notes.append(p)
    return sorted(notes, key=lambda p: p.stem)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--mode", choices=["full", "chunk"], default="full")
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--segment-chars", type=int, default=1600)
    ap.add_argument("--context-format", choices=["json", "blocks"], default="json")
    ap.add_argument("--raw-max-items", type=int, default=8)
    ap.add_argument("--raw-format", choices=["json", "blocks", "final-blocks"], default="json")
    ap.add_argument("--normalize-max-items", type=int, default=10)
    ap.add_argument("--placement-mode", choices=["list", "item", "batch", "agentic"], default="list")
    ap.add_argument("--placement-batch-size", type=int, default=4)
    ap.add_argument("--placement-repair-mode", choices=["none", "prompt", "agentic"], default="prompt")
    ap.add_argument("--placement-vault", default="")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = [
        run_note(
            n, out_root, args.model, args.base_url, args.mode, args.max_tokens,
            args.segment_chars, args.raw_max_items, args.normalize_max_items,
            args.placement_mode, args.placement_batch_size, args.raw_format,
            args.context_format, args.placement_vault, args.placement_repair_mode
        )
        for n in journal_notes(args.notes)
    ]
    report = {"notes": len(results), "mode": args.mode, "seconds": elapsed(t0), "results": results}
    (out_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
