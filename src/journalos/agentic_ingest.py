#!/usr/bin/env python3
"""Agentic ingest harness for experimental tool-driven wiki writing.

This gives the model a small tool loop: read_file, list_dir, grep, write_file,
and edit_file. The staged ingest path is the default public workflow; this file
is kept as an experimental backend.

Portable protocol: the model returns ONE JSON action object per turn in its
message content (works whether or not the server supports native tool-calling).

SAFETY: every file op is confined to --vault.

  # serve a model (any OpenAI-compatible endpoint), then:
  python src/journalos/agentic_ingest.py --vault /tmp/journalos-sandbox \
      --note Journal/2026-06-24.md --model <id> --base-url http://127.0.0.1:8090/v1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROTECTED_VAULT = os.environ.get("JOURNALOS_PROTECTED_VAULT", "")
MAX_OBS = 2400  # cap each observation so the trajectory fits context

# _about.md files state each directory's PURPOSE and are a stable contract DURING a build:
# locking them stops mid-ingest scope drift / scope-creep (an agent "solving" an awkward
# placement by quietly broadening a directory). A later directory-reorg pass flips this to
# True to grant edit privilege. write_file can still CREATE a new dir's _about.md (the
# no-overwrite guard prevents clobbering an existing one).
ALLOW_ABOUT_EDITS = False

SYSTEM = """You are JournalOS, ingesting one journal note into a personal wiki. Work \
agentically with tools until the wiki reflects the note, then finish.

Workflow you MUST follow:
1. To record a fact, FIRST find the existing page it belongs on (use grep / list_dir /
   read_file). The catalog of existing pages is given below — do NOT invent a new page
   when one already exists (e.g. the local-LLM project already has a page; find it).
2. Add a fact with append_section — a newest-first dated bullet under a "## section".
   That is the normal way to record something.
3. Use edit_file only to FIX existing text. NEVER use write_file on a page that exists.
4. Use write_file ONLY to create a brand-new page that does not exist yet (e.g. a new
   person's stub).

Rules:
- Ground every claim in the note; never invent facts. Cite with ([[YYYY-MM-DD]]).
- Capture every named person (a stub page minimum). Update trackers when warranted.

Placement — file each fact where it belongs by the page's PURPOSE:
- a workout/exercise -> Wiki/Health/Progressive Overload Tracker.md
- a problem to solve -> Wiki/Trackers/Problems.md
- a fact/event about a person or pet -> their Wiki/Profiles/<Name>.md
- the day's one-line summary -> Wiki/Trackers/Daily State.md
Cover EVERY distinct fact in the note before you finish — do not stop after one or two.

Each turn output EXACTLY ONE JSON object and nothing else:
  {"thought": "...", "action": "<name>", "args": { ... }}
Keep "thought" to ONE short sentence (<=15 words). Never paste tool output into it —
long thoughts get truncated and break the JSON.
Tools:
  list_dir       {"path": "Wiki/Projects"}             -> entries
  read_file      {"path": "Wiki/index.md"}             -> contents (truncated)
  grep           {"pattern": "Needle", "path": "Wiki"}  -> matching file:line
  append_section {"path": "...", "section": "...", "text": "..."} -> prepend a dated
                 bullet under a "## section" (SAFE, non-destructive — prefer this)
  edit_file      {"path": "...", "old": "...", "new": "..."} -> replace unique text
  write_file     {"path": "...", "content": "..."}     -> create a NEW file ONLY
  finish         {"summary": "..."}                    -> stop
Start by exploring to find the right existing pages."""


def safe(vault: Path, rel: str) -> Path:
    p = (vault / rel).resolve()
    if not str(p).startswith(str(vault.resolve())):
        raise ValueError("path escapes the sandbox")
    return p


# Models pad a NEW file with an uncited filler bullet ("* Initial entry.") — junk that also
# breaks the cite-every-claim rule. Strip these placeholder bullets deterministically.
_PLACEHOLDERS = {"initial entry", "initial entries", "initial", "placeholder", "tbd",
                 "n/a", "none", "no entries yet", "first entry"}


def is_placeholder(line: str) -> bool:
    core = line.strip().lstrip("-*•").strip().rstrip(".:;").strip().lower()
    return core in _PLACEHOLDERS


def strip_placeholders(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not is_placeholder(l))


def tool(vault: Path, action: str, args: dict, touched: set, pages: list) -> str:
    import difflib
    try:
        if action in ("write_file", "append_section", "edit_file"):
            rel = args.get("path", "")
            if not (rel.startswith("Wiki/") and rel.endswith(".md")):
                return ("ERROR: 'path' must be a wiki page under Wiki/ ending in .md, e.g. "
                        "'Wiki/Projects/Knowledge System.md'. Copy an EXACT path from the catalog.")
            if len(rel.split("/")) < 3:  # enforce Wiki/<dir>/<name>.md — no bare top-level files
                return ("ERROR: content files must live INSIDE a directory: 'Wiki/<dir>/<name>.md', "
                        "not a bare 'Wiki/<name>.md'. Put it under the directory whose purpose "
                        "fits (create a new described directory if none does).")
            if (action in ("append_section", "edit_file") and Path(rel).name == "_about.md"
                    and not ALLOW_ABOUT_EDITS):
                return ("ERROR: _about.md states this directory's PURPOSE and is LOCKED during "
                        "ingest. Do not change a directory's scope to fit a fact — file the fact "
                        "in a content file under whichever directory's purpose already fits.")
        if action == "list_dir":
            d = safe(vault, args.get("path", "."))
            return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())[:80])
        if action == "read_file":
            p = safe(vault, args["path"])
            return p.read_text(encoding="utf-8")[:MAX_OBS] if p.exists() else "(no such file)"
        if action == "grep":
            r = subprocess.run(["grep", "-rni", args.get("pattern", ""),
                                str(safe(vault, args.get("path", "Wiki")))],
                               capture_output=True, text=True, timeout=20)
            lines = [l.replace(str(vault) + "/", "") for l in r.stdout.splitlines()][:30]
            return "\n".join(lines) or "(no matches)"
        if action == "write_file":
            p = safe(vault, args["path"])
            if p.exists():  # never overwrite — the destructive failure mode
                return ("ERROR: that file already EXISTS. Use append_section to add a "
                        "fact, or edit_file to change text. Do not overwrite.")
            stem = Path(args["path"]).stem.lower()  # dedup nudge: same-name page elsewhere?
            twin = next((pp for pp in pages if difflib.SequenceMatcher(
                None, stem, Path(pp).stem.lower()).ratio() >= 0.9), None)
            if twin:
                return (f"STOP: a page for this already exists at '{twin}'. Use "
                        f"append_section on it instead of creating a duplicate.")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(strip_placeholders(args.get("content", "")), encoding="utf-8")
            touched.add(args["path"]); return f"created new file {args['path']}"
        if action == "append_section":
            p = safe(vault, args["path"])
            if not p.exists():
                return "ERROR: file does not exist — create it with write_file first"
            text = (args.get("text") or "").strip().lstrip("-•*").strip()  # avoid '- •' double bullet
            if not text:
                return "ERROR: empty text"
            if is_placeholder(text):
                return "ERROR: refusing a placeholder bullet — add a grounded, cited fact only."
            bullet = "- " + text
            lines = p.read_text(encoding="utf-8").splitlines()
            section = args.get("section", "")
            idx = next((i for i, l in enumerate(lines)
                        if l.startswith("## ") and l[3:].strip() == section), None)
            if idx is None:  # fall back to just after the H1 title
                idx = next((i for i, l in enumerate(lines) if l.startswith("# ")), 0)
            ins = idx + 1
            while ins < len(lines) and lines[ins].strip() == "":
                ins += 1
            lines.insert(ins, bullet)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            touched.add(args["path"]); return f"appended to {args['path']} (section: {section or 'top'})"
        if action == "edit_file":
            p = safe(vault, args["path"])
            if not p.exists():
                return "ERROR: no such file"
            text = p.read_text(encoding="utf-8")
            old, new = args.get("old", ""), args.get("new", "")
            if old not in text:
                return ("ERROR: 'old' text not found — read_file first and copy the exact "
                        f"text verbatim (the file has {len(text.splitlines())} lines).")
            if text.count(old) > 1:
                return "ERROR: 'old' text is not unique — include more surrounding context"
            p.write_text(text.replace(old, new), encoding="utf-8")
            touched.add(args["path"]); return f"edited {args['path']}"
        return f"ERROR: unknown action '{action}'"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def parse_action(text: str) -> dict | None:
    s = text.strip()
    if "```" in s and s.count("```") >= 2:
        s = s.split("```")[1]
        s = s[4:] if s.lstrip().lower().startswith("json") else s
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        d = json.loads(s[i:j + 1])
        return d if isinstance(d, dict) and "action" in d else None
    except json.JSONDecodeError:
        return None


def final_content(content: str) -> str:
    """Drop Gemma thinking blocks before parsing or adding assistant history."""
    marker = "<channel|>"
    if marker in content:
        return content.rsplit(marker, 1)[-1].strip()
    return content


def with_think_trigger(messages: list) -> list:
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            text = str(m.get("content", ""))
            if not text.startswith("<|think|>"):
                m["content"] = "<|think|>\n" + text
            return out
    return [{"role": "system", "content": "<|think|>"}] + out


def chat(base_url: str, model: str, messages: list, max_tokens: int,
         reasoning_budget: int = 0, timeout: int = 600) -> tuple[str, int]:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 1.0, "top_p": 0.95, "top_k": 64}
    if reasoning_budget > 0:
        payload.update({
            "messages": with_think_trigger(messages),
            "max_tokens": max_tokens + reasoning_budget,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "repeat_penalty": 1.15,
            "repetition_penalty": 1.15,
            "frequency_penalty": 0.2,
            "thinking_budget": reasoning_budget,
            "thinking_start_token": "<|channel>",
            "thinking_end_token": "<channel|>",
        })
    else:
        # reasoning models (Qwen3.5) default to thinking, which leaves content empty
        # in a tool loop — force it off so we get direct JSON actions.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    msg = final_content(data["choices"][0]["message"].get("content") or "")
    toks = (data.get("usage") or {}).get("total_tokens", 0)
    return msg, toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--trajectory", default="", help="Optional path to dump the full trajectory.")
    args = ap.parse_args()

    vault = Path(args.vault).resolve()
    if PROTECTED_VAULT and str(vault) == str(Path(PROTECTED_VAULT).expanduser().resolve()):
        print("REFUSING: --vault matches JOURNALOS_PROTECTED_VAULT.", file=sys.stderr); return 2
    note = vault / args.note
    if not note.exists():
        print(f"note not found: {note}", file=sys.stderr); return 1

    paths = sorted(p.relative_to(vault).as_posix()
                   for d in ("Projects", "Trackers", "Profiles", "Self", "Health", "Finance", "Ideas")
                   for p in (vault / "Wiki" / d).glob("*.md"))
    catalog = "\n".join(paths)
    task = (f"Ingest this note: {args.note}\n\n--- NOTE ---\n{note.read_text(encoding='utf-8')}\n"
            f"--- END NOTE ---\n\nEXISTING pages — to update one, copy its EXACT path from "
            f"this list (do NOT invent paths or new pages when one fits):\n{catalog}\n\n"
            f"Begin by deciding which of these existing pages this note affects.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]

    touched: set = set()
    tool_counts: dict = {}
    trajectory = []
    t0 = time.perf_counter()
    total_tokens = 0
    finished = False
    bad = 0

    for step in range(args.max_steps):
        try:
            content, toks = chat(args.base_url, args.model, messages, args.max_tokens)
        except Exception as e:  # noqa: BLE001
            trajectory.append({"step": step, "error": f"chat failed: {e}"}); break
        total_tokens += toks
        act = parse_action(content)
        if not act:
            bad += 1
            messages.append({"role": "assistant", "content": content[:1000]})
            messages.append({"role": "user", "content": "Reply with ONE valid JSON action object only."})
            trajectory.append({"step": step, "unparseable": content[:200]})
            if bad >= 4:
                break
            continue
        action = act.get("action"); a_args = act.get("args", {}) if isinstance(act.get("args"), dict) else {}
        tool_counts[action] = tool_counts.get(action, 0) + 1
        if action == "finish":
            finished = True
            trajectory.append({"step": step, "action": "finish", "summary": a_args.get("summary", "")})
            break
        obs = tool(vault, action, a_args, touched, paths)
        trajectory.append({"step": step, "action": action, "args": a_args, "obs": obs[:300]})
        messages.append({"role": "assistant", "content": content[:1500]})
        messages.append({"role": "user", "content": f"OBSERVATION:\n{obs[:MAX_OBS]}"})

    summary = {
        "model": args.model, "note": args.note, "finished": finished,
        "steps": len(trajectory), "tool_counts": tool_counts,
        "files_touched": sorted(touched), "unparseable_replies": bad,
        "total_tokens": total_tokens, "seconds": round(time.perf_counter() - t0),
    }
    if args.trajectory:
        Path(args.trajectory).write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
