#!/usr/bin/env python3
"""Run the staged JournalOS ingest pipeline and compile it into a wiki.

This is the write-capable counterpart to ingest_probe.py. It keeps the
LLM responsible for extraction, normalization, and semantic placement, while the
harness handles structural wiki mechanics:

- write primary placements first
- write profile side effects second
- create missing pages safely inside a sandbox vault
- log stage outputs for review
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_probe as probe

PROTECTED_VAULT = os.environ.get("JOURNALOS_PROTECTED_VAULT", "")
PILLARS_PATH = Path(__file__).resolve().parent / "scaffold" / "pillars.json"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 1)


def safe_path(vault: Path, rel: str) -> Path:
    rel = str(rel or "").strip()
    if not rel.startswith("Wiki/") or not rel.endswith(".md"):
        raise ValueError(f"unsafe wiki path: {rel}")
    path = (vault / rel).resolve()
    if not str(path).startswith(str(vault.resolve()) + "/"):
        raise ValueError(f"path escapes vault: {rel}")
    return path


def title_from_path(path: Path) -> str:
    title = path.stem.replace("-", " ").replace("_", " ").strip()
    return re.sub(r"\s+", " ", title).title() or "Untitled"


def ensure_page(path: Path, heading: str = "") -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    title = heading.strip() or title_from_path(path)
    path.write_text(f"# {title}\n\n## Facts\n", encoding="utf-8")
    return True


def ensure_scaffold(vault: Path) -> None:
    wiki = vault / "Wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    index = wiki / "Index.md"
    if not index.exists():
        index.write_text("# JournalOS Wiki\n\n", encoding="utf-8")
    data = json.loads(PILLARS_PATH.read_text(encoding="utf-8"))
    for pillar in data.get("pillars", []):
        name = pillar.get("name", "")
        if not name:
            continue
        slug = probe.slug(name)
        d = wiki / slug
        d.mkdir(parents=True, exist_ok=True)
        idx = d / "Index.md"
        if not idx.exists():
            purpose = pillar.get("purpose", "")
            boundary = pillar.get("boundary", "")
            idx.write_text(
                f"# {name}\n\nPurpose: {purpose}\n\nBoundary: {boundary}\n",
                encoding="utf-8",
            )
    profiles = wiki / "social-connections" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    ensure_page(profiles / "Index.md", "Profiles")


def append_under_section(path: Path, section: str, bullet: str) -> bool:
    section = (section or "Facts").strip() or "Facts"
    bullet = str(bullet or "").strip()
    if not bullet:
        return False
    if not bullet.startswith("- "):
        bullet = "- " + bullet.lstrip("-* ").strip()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    header = f"## {section}"
    if header not in lines:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([header])
    idx = lines.index(header)
    insert_at = idx + 1
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1
    if bullet in lines:
        return False
    lines.insert(insert_at, bullet)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return True


def compile_wiki(vault: Path, date: str, placements_text: str) -> dict:
    data = probe.parse_json_object(placements_text)
    result = {
        "date": date,
        "primary_written": 0,
        "side_effects_written": 0,
        "pages_created": [],
        "pages_touched": [],
        "skipped": [],
    }
    touched: set[str] = set()

    for item in data.get("primary_placements", []) if isinstance(data.get("primary_placements"), list) else []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("primary_target") or item.get("target_hint") or "")
        fact = str(item.get("fact") or "").strip()
        citation = str(item.get("citation") or f"[[{date}]]").strip()
        if not target or not fact:
            result["skipped"].append({"kind": "primary", "reason": "missing target or fact", "item": item})
            continue
        try:
            path = safe_path(vault, target)
        except ValueError as exc:
            result["skipped"].append({"kind": "primary", "reason": str(exc), "item_id": item.get("item_id", "")})
            continue
        if ensure_page(path):
            result["pages_created"].append(path.relative_to(vault).as_posix())
        bullet = fact if "[[" in fact else f"{fact} ({citation})"
        if append_under_section(path, str(item.get("section") or "Facts"), bullet):
            result["primary_written"] += 1
            touched.add(path.relative_to(vault).as_posix())

    for effect in data.get("profile_side_effects", []) if isinstance(data.get("profile_side_effects"), list) else []:
        if not isinstance(effect, dict):
            continue
        path_text = str(effect.get("path") or "")
        fact = str(effect.get("fact") or "").strip()
        if not path_text or not fact:
            result["skipped"].append({"kind": "profile_side_effect", "reason": "missing path or fact", "item": effect})
            continue
        try:
            path = safe_path(vault, path_text)
        except ValueError as exc:
            result["skipped"].append({"kind": "profile_side_effect", "reason": str(exc), "item_id": effect.get("item_id", "")})
            continue
        if ensure_page(path, str(effect.get("name") or "")):
            result["pages_created"].append(path.relative_to(vault).as_posix())
        if append_under_section(path, "Facts", fact):
            result["side_effects_written"] += 1
            touched.add(path.relative_to(vault).as_posix())

    result["pages_touched"] = sorted(touched)
    result["pages_created"] = sorted(set(result["pages_created"]))
    return result


def diagnostic_count(warnings: list[str]) -> int:
    return sum("exact source quote" in warning for warning in warnings)


def blocking_warnings(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if "exact source quote" not in warning]


def run_note(note: Path, vault: Path, out_root: Path, args: argparse.Namespace) -> dict:
    probe_result = probe.run_note(
        note, out_root, args.model, args.base_url, args.mode, args.max_tokens,
        args.segment_chars, args.raw_max_items, args.normalize_max_items,
        args.placement_mode, args.placement_batch_size, args.raw_format,
        args.context_format, str(vault), args.placement_repair_mode
    )
    date = note.stem
    out_dir = out_root / date
    placements = (out_dir / "04-placement-merged.json").read_text(encoding="utf-8")
    compile_result = compile_wiki(vault, date, placements)
    (out_dir / "06-compile.json").write_text(json.dumps(compile_result, indent=2) + "\n", encoding="utf-8")

    warnings = list(probe_result.get("warnings", []))
    probe_result["diagnostics"] = diagnostic_count(warnings)
    probe_result["blocking_warnings"] = blocking_warnings(warnings)
    probe_result["compile"] = compile_result
    return probe_result


def copy_notes(notes: list[Path], vault: Path) -> None:
    journal = vault / "Journal"
    journal.mkdir(parents=True, exist_ok=True)
    for note in notes:
        shutil.copy2(note, journal / note.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", nargs="+", required=True)
    ap.add_argument("--vault", required=True)
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
    ap.add_argument("--reset-vault", action="store_true")
    args = ap.parse_args()

    vault = Path(args.vault).expanduser().resolve()
    if PROTECTED_VAULT and vault == Path(PROTECTED_VAULT).expanduser().resolve():
        print("REFUSING: --vault matches JOURNALOS_PROTECTED_VAULT.", file=sys.stderr)
        return 2
    if args.reset_vault and vault.exists():
        shutil.rmtree(vault)
    vault.mkdir(parents=True, exist_ok=True)
    ensure_scaffold(vault)

    notes = probe.journal_notes(args.notes)
    copy_notes(notes, vault)
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = [run_note(note, vault, out_root, args) for note in notes]
    report = {
        "notes": len(results),
        "mode": args.mode,
        "seconds": elapsed(t0),
        "vault": str(vault),
        "out": str(out_root),
        "results": results,
        "totals": {
            "primary_written": sum(r["compile"]["primary_written"] for r in results),
            "side_effects_written": sum(r["compile"]["side_effects_written"] for r in results),
            "diagnostics": sum(r.get("diagnostics", 0) for r in results),
            "blocking_warnings": sum(len(r.get("blocking_warnings", [])) for r in results),
            "skipped": sum(len(r["compile"]["skipped"]) for r in results),
        },
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
