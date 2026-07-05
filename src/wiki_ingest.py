#!/usr/bin/env python3
"""JournalOS wiki ingest: bare-minimum pass, people and pets only.

Design: the harness owns every structural decision (paths, sections,
citations, dedup, the index), and the model only answers small questions:

  1. roster  (1 call)            who is mentioned in the note? people vs pets
  2. facts   (1 call/paragraph)  every durable fact in ONE paragraph, about
                                  ANY entity (self or a named person/pet)
  3. sanity  (1 call/new entity) is this NEW page actually a living person or
                                  pet, not a project/tool/AI talked about in a
                                  personified way? Only runs on pages created
                                  this note, never on already-established ones

Stage 2 used to ask about one entity against the WHOLE note. That avoided
pronoun ambiguity but caused a worse failure: shown the whole note, the
model would see a name mentioned once near the top and then attribute an
unrelated, unnamed paragraph several paragraphs later (e.g. the journal
author's own first-person workout log) to that person, since nothing else
in the note competed for its attention. Chunking by paragraph and asking
about all entities at once fixes this structurally: a paragraph that never
mentions "Mukul" is never shown to the model in a call where "Mukul" is
even a possible answer, because the model can't see it.

There used to be a fact-level "verify" pass here that re-read every newly
written bullet against the whole note before keeping it. It was dropped:
once stage 2 moved to per-paragraph chunking, verify's remaining job
(catching two co-mentioned people getting mixed up within one paragraph)
became rare, while its false-positive rate did not; it kept discarding
correct facts with self-contradictory reasoning (e.g. stating "the journal
confirms this is self" and then answering "correct: no" anyway). It also
roughly quadrupled runtime.

Stage 3 (the current sanity check) is a narrower, cheaper descendant of
that idea: rather than trying to make the roster prompt perfectly exclude
every project/tool/AI assistant up front, which turned into whack-a-mole
against specific names (Claude, Pi, LiOS, ...) that would never fully
generalize to whatever the author names their next project. Instead, the
roster prompt stays permissive, and this stage catches the rare case once, per
NEW entity, instead of re-litigating every fact.

Everything the model returns is validated in Python (the evidence quote
must actually appear in the paragraph it was drawn from, and the subject
must be "self" or a name from the stage-1 roster). Facts that fail
validation go to wiki/Quarantine.md instead of being guessed at. Writes
are idempotent: re-running a note never duplicates bullets, so a crashed
run can simply be re-run.

Out of scope for this pass (deliberately): topic/pillar pages, trackers,
routing. Those return in a later pass once people/pets are solid.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MAX_FACTS_PER_ENTITY = 15


@dataclass
class Settings:
    """Every model-facing knob, in one place. Defaults are calibrated for
    Gemma 4 E4B (4-bit); the ingest CLI overrides them from config.yaml when
    that file is present. See config.yaml for what each field controls."""

    model: str = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
    base_url: str = "http://127.0.0.1:8090/v1"
    temperature: float = 1.0
    timeout: int = 240        # seconds to wait on a single model call
    attempts: int = 3         # retries before a call is considered failed
    tokens_roster: int = 600      # stage 1
    tokens_facts: int = 1800      # stage 2, per paragraph
    tokens_sanity: int = 300      # stage 3, per new page
    tokens_summary: int = 2048    # stage 4, reasoning attempt
    tokens_summary_fallback: int = 400   # stage 4, non-reasoning fallback
    summary_window_days: int = 60        # only summarize facts this recent
    summary_use_reasoning: bool = True   # think-then-answer summary prompt


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------
# Model access
# --------------------------------------------------------------------------

def chat(base_url: str, model: str, prompt: str, *, max_tokens: int,
         temperature: float = 1.0, timeout: int = 240, attempts: int = 3) -> str:
    """One model call with retries. Raises RuntimeError if all attempts fail.

    Sampling settings follow the model vendor's recommendation for Gemma.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            return (data["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001 - network/parse errors alike
            last_error = exc
            if attempt < attempts:
                log(f"   model call failed ({exc}), retrying in {5 * attempt}s")
                time.sleep(5 * attempt)
    raise RuntimeError(f"model call failed after {attempts} attempts: {last_error}")


class StageLog:
    """Runs model calls, optionally recording each prompt/output pair under
    logs/<date>/. Off by default; if a run goes wrong, re-run it with
    logging enabled (the ingest CLI's --log flag) to get the full
    per-stage trail needed to debug it."""

    def __init__(self, out_dir: Path, settings: "Settings", enabled: bool = False):
        self.out_dir = out_dir
        self.settings = settings
        self.enabled = enabled

    def call(self, stage: str, prompt: str, *, max_tokens: int) -> str:
        s = self.settings
        t0 = time.perf_counter()
        output = chat(s.base_url, s.model, prompt, max_tokens=max_tokens,
                      temperature=s.temperature, timeout=s.timeout, attempts=s.attempts)
        seconds = round(time.perf_counter() - t0, 1)
        if self.enabled:
            stage_dir = self.out_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (stage_dir / "output.txt").write_text(output + "\n", encoding="utf-8")
            (stage_dir / "meta.json").write_text(
                json.dumps({"seconds": seconds}) + "\n", encoding="utf-8")
        log(f"   {stage}: {seconds}s")
        return output


# --------------------------------------------------------------------------
# Block parsing (the model always answers in plain-text blocks, never JSON)
# --------------------------------------------------------------------------

def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip().strip('"')
    return fields


def parse_labelled_blocks(text: str, labels: tuple[str, ...]) -> list[dict]:
    """Split model output on lines that start with one of the labels.

    A label followed by a colon (e.g. the "fact:" field line) is a field,
    not a block delimiter, so it must not match here.
    """
    pattern = r"(?im)^\s*(" + "|".join(labels) + r")\b(?!\s*:).*$"
    parts = re.split(pattern, text)
    blocks: list[dict] = []
    for index in range(1, len(parts) - 1, 2):
        blocks.append(parse_fields(parts[index + 1]))
    return blocks


# --------------------------------------------------------------------------
# Stage 1: roster. Who is in this note?
# --------------------------------------------------------------------------

def roster_prompt(date: str, note_text: str) -> str:
    return (
        f"List the people and pets mentioned in this journal note dated {date}.\n"
        "The journal author (\"I\", \"me\", \"self\") is not a person to list.\n\n"
        "Write one plain-text block per person or pet, exactly in this shape:\n\n"
        "PERSON\n"
        "name: their name\n"
        "evidence: exact quote mentioning them\n\n"
        "PET\n"
        "name: the pet's name\n"
        "evidence: exact quote mentioning them\n\n"
        "If unsure whether a name is a pet or a person, use PERSON.\n"
        "If nobody is mentioned, reply with the single word NONE.\n\n"
        f"--- NOTE ---\n{note_text}\n--- END NOTE ---\n"
    )


def parse_roster(text: str) -> dict[str, list[str]]:
    """Parse PERSON/PET blocks. Split here directly (instead of reusing
    parse_labelled_blocks) since we need to know which of the two labels
    matched for each block, not just its fields."""
    roster: dict[str, list[str]] = {"people": [], "pets": []}
    seen: set[str] = set()
    pattern = r"(?im)^\s*(PERSON|PET)\b.*$"
    parts = re.split(pattern, text)
    for index in range(1, len(parts) - 1, 2):
        label = parts[index].strip().upper()
        fields = parse_fields(parts[index + 1])
        name = re.sub(r"\s+", " ", fields.get("name", "")).strip()
        if not name or name.lower() in ("self", "none") or name.lower() in seen:
            continue
        seen.add(name.lower())
        roster["pets" if label == "PET" else "people"].append(name)
    return roster


# --------------------------------------------------------------------------
# Stage 2: facts about all entities, one paragraph at a time.
#
# Earlier version of this stage asked about ONE entity against the WHOLE
# note. That fixed pronoun ambiguity but introduced a worse failure: shown
# the whole note, the model would see a name mentioned once near the top
# and then attribute an unrelated, unnamed paragraph several paragraphs
# later (e.g. the journal author's own first-person workout log) to that
# person, since nothing else in the note competed for its attention.
#
# Chunking by paragraph and asking about ALL entities at once per chunk
# fixes this structurally rather than by catching it after the fact: a
# paragraph that never mentions "Mukul" is never shown to the model in a
# call where "Mukul" is even a possible answer to attribute it to, because
# the model can't see it. This also cuts the number of stage-2 calls from
# (1 + number of entities) down to (number of paragraphs), which is
# usually fewer.
# --------------------------------------------------------------------------

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_facts_prompt(date: str, roster: dict, passage: str) -> str:
    names = ["self"] + roster["people"] + roster["pets"]
    listing = ", ".join(f'"{name}"' for name in names)
    return (
        f"Below is one passage from a longer journal note dated {date} (you are not "
        f"shown the rest of the note). List every durable fact this passage contains "
        f"about any of these entities: {listing}. \"self\" means the journal author "
        "(\"I\", \"me\").\n\n"
        "Write one plain-text block per fact, exactly in this shape:\n\n"
        "FACT\n"
        f"subject: exactly one of {listing}\n"
        "fact: one short self-contained sentence, naming the subject explicitly\n"
        "evidence: exact quote copied word-for-word from the PASSAGE\n\n"
        "Rules:\n"
        f"- At most {MAX_FACTS_PER_ENTITY} FACT blocks. Use as many as the passage "
        "actually supports -- do not pad and do not skip real facts to stay under "
        "the limit.\n"
        "- subject must be exactly one of the names listed above, never a pronoun "
        "and never a name that is not in that list.\n"
        "- Base the subject only on THIS passage. Do not assume anything about who "
        "is doing what based on names mentioned only in other parts of the note you "
        "have not been shown -- if this passage alone does not make the subject "
        "clear, skip that fact rather than guessing.\n"
        "- If one sentence names several entities as joint subjects of the same "
        "thing (e.g. \"Shubhi, Rash, Mukul and me are doing the challenge\"), write "
        "ONE SEPARATE FACT block per named entity, each with its own subject line -- "
        "do not merge them into a single fact and do not write a fact whose subject "
        "line lists more than one name.\n"
        "- Durable facts only: things worth remembering next month (traits, habits,\n"
        "  measurements, relationships, preferences, decisions, events).\n"
        "- evidence must be an exact quote from the PASSAGE below, not a paraphrase.\n"
        "- If this passage has no durable facts about any listed entity, reply with "
        "the single word NONE.\n\n"
        f"--- PASSAGE ---\n{passage}\n--- END PASSAGE ---\n"
    )


def normalize_for_match(text: str) -> str:
    """Normalize quotes/whitespace/case so evidence checks tolerate copy drift."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_chunk_fact(fields: dict, roster: dict, passage: str) -> tuple[str, str, str]:
    """Return (fact_text, canonical_subject, "") if valid, else ("", "", reason)."""
    fact_text = fields.get("fact", "").strip()
    if not fact_text:
        return "", "", "missing fact text"

    subject_raw = fields.get("subject", "").strip()
    subject_lower = subject_raw.lower()
    canonical = "self" if subject_lower == "self" else ""
    if not canonical:
        for name in roster["people"] + roster["pets"]:
            if name.lower() == subject_lower:
                canonical = name
                break
    if not canonical:
        return "", "", f"unknown subject '{subject_raw}'"

    evidence = fields.get("evidence", "").strip()
    if not evidence:
        return "", "", "missing evidence"
    if normalize_for_match(evidence) not in normalize_for_match(passage):
        return "", "", "evidence is not a quote from the passage"
    return fact_text, canonical, ""


# --------------------------------------------------------------------------
# Stage 3: sanity-check newly created entity pages.
#
# The roster prompt above is deliberately permissive again. It does not
# try to exclude projects, tools, or AI assistants. Chasing that at the
# roster stage turned into a whack-a-mole of specific examples (Claude, Pi,
# LiOS, ...) that would never fully generalize to whatever the author names
# their next project. Instead, catch it here, once per NEW entity page
# rather than once per fact: this is a much smaller and more reliable job
# than the fact-attribution "verify" stage that was dropped earlier (which
# had to judge dozens of facts against the whole note every day; this only
# has to judge the handful of genuinely new names that show up).
# --------------------------------------------------------------------------

def entity_sanity_prompt(name: str, bullets: list[str]) -> str:
    facts = "\n".join(f"- {bullet}" for bullet in bullets)
    return (
        f"A new wiki page was just created for \"{name}\", based on these facts "
        f"pulled from a journal note:\n{facts}\n\n"
        f"Is \"{name}\" a real living person or pet (an animal), as opposed to a "
        "project, app, tool, AI assistant or model, company, or other named thing? "
        "Journals sometimes describe a tool or project in a personified way (e.g. "
        "\"brainstormed with Codex\", \"my project LiOS\"), which can look like a "
        "person from the facts alone -- watch for that.\n\n"
        "Reply with exactly this shape:\n"
        "ANSWER\n"
        "is_living_being: yes | no\n"
        "reason: short reason\n"
    )


def parse_entity_sanity(text: str) -> tuple[bool, str]:
    for fields in parse_labelled_blocks(text, ("ANSWER",)):
        value = fields.get("is_living_being", "").strip().lower()
        return value.startswith("y"), fields.get("reason", "").strip()
    return True, "no answer parsed, defaulting to keep"


# --------------------------------------------------------------------------
# Stage 4: recency-weighted summary. Runs once per entity that got new
# facts this note, after stage 3 (so a page stage 3 just deleted never
# gets a summary written for it). Bullets are sorted newest-journal-date
# first (see sort_bullets_by_date), then narrowed to the last
# summary_window_days days (see facts_within_days) before being shown --
# a time window rather than a fixed fact count, so a quiet entity's
# summary isn't padded out with things from months ago just to hit a
# count, and a chatty entity's summary isn't cut off mid-week.
# --------------------------------------------------------------------------

def humanize_self_references(text: str) -> str:
    """Replace the literal placeholder "self" with an unambiguous phrase
    before a fact is shown to a prompt writing prose about someone ELSE.
    Left as-is, a fact like "Chiranjib met with self today" reads to an
    ordinary English parser as a reflexive pronoun ("Chiranjib met with
    himself"), silently losing the fact that a third person (the journal
    author) was involved. "self" is never actually a pronoun here, it's
    this pipeline's placeholder name for the author, so swap it for a
    phrase that can't be mis-parsed that way."""
    return re.sub(r"\bself\b", "the journal author", text, flags=re.IGNORECASE)


def summary_prompt(name: str, today: str, bullets: list[str], reasoning: bool = False,
                   window_days: int = 60) -> str:
    subject = "the journal author (self)" if name == "self" else name
    facts = "\n".join(f"- {bullet}" for bullet in facts_within_days(bullets, today, window_days))
    if reasoning:
        return (
            f"Here are known facts about {subject}, newest first, each with the date "
            f"it was journaled. Today's date is {today}.\n\n{facts}\n\n"
            "First, think it through: what are the distinct topics/threads here (e.g. "
            "job, health, a relationship, a project)? For each thread, what is the "
            "MOST RECENT state, based on the dates -- does a newer fact update or "
            "contradict an older one in that same thread?\n\n"
            "Then, on its own line, write:\n"
            "FINAL SUMMARY: <2-4 sentence prose summary, no heading, no bullet points>\n"
            f"That summary must describe the CURRENT state of {subject}, reflecting "
            "only the most recent state of each thread you identified above -- never "
            "state a thread's stale older state as if it were still true.\n"
        )
    return (
        f"Here are known facts about {subject}, newest first, each with the date "
        f"it was journaled. Today's date is {today}.\n\n{facts}\n\n"
        f"Write a short summary (2-4 sentences) of the CURRENT state of {subject}, "
        "in plain prose, no heading and no bullet points. Weight recent facts much "
        "more heavily than older ones: if a newer fact updates, changes, or "
        "contradicts an older one (e.g. a job change, a status change, a "
        "relationship change), reflect the newer state and do not also state the "
        "stale older one as if it were still true. Durable traits or preferences "
        "that nothing more recent has contradicted are still fine to include.\n"
    )


def extract_final_summary(reply: str) -> str:
    """Pull the actual answer out of a reasoning reply.

    Matching a specific marker phrase turned out not to be robust: the
    model doesn't reliably say "FINAL SUMMARY" verbatim; it's also been
    seen writing "Final Output Generation" instead, and a regex trying to
    skip past reasoning-step headings like "4. **Construct Final
    Summary:**" runs into a worse bug than the one it fixes: a greedy
    ".+" capture on that FIRST occurrence swallows the rest of the string,
    including the real marker further down, so "take the last match"
    doesn't actually help once the first match has already eaten
    everything after it.

    What's actually reliable across every case seen: the real answer is
    the LAST paragraph of the reply, regardless of what heading (if any)
    introduces it. So split on blank lines and take the last one, then
    strip a "final summary:" style prefix off it if the model did include
    one. The model sometimes puts a markdown horizontal rule ("---") on
    its own line directly above "FINAL SUMMARY:" with no blank line in
    between, so the two end up as one paragraph and the marker isn't at
    the very start of it. Strip a leading rule line first, or the
    "^...final summary" anchor never matches and the marker leaks through.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", reply.strip()) if p.strip()]
    last = paragraphs[-1] if paragraphs else reply
    last = re.sub(r"^(?:[-*_]{3,}\s*\n)+", "", last)
    last = re.sub(r"^\**\s*final summary\s*:?\**\s*", "", last, flags=re.IGNORECASE)
    return strip_summary_wrapper(last)


def looks_like_unfinished_summary(text: str) -> bool:
    """True if `text` looks like a truncated reasoning fragment rather
    than a finished summary, e.g. the model ran out of max_tokens mid-
    thought and never actually reached its final answer. Used to trigger
    a fallback rather than writing garbage to a page.

    A real 2-4 sentence summary ends in terminal punctuation and has more
    than a handful of words. A fragment cut off mid-reasoning (things
    like "(Final Output Generation)", "**Final Output Generation:", or
    even just "Final Output Generation." once the model burns its whole
    token budget on reasoning and never gets to the actual answer) can
    slip past a punctuation-only check by coincidentally ending in a
    period, so a minimum word count is checked too."""
    if not text or len(text) < 15:
        return True
    if text.rstrip().endswith((":", "-", "*")):
        return True
    if re.match(r"^(thought process|\d+[.)]|\*\*|\()", text, re.IGNORECASE):
        return True
    if not re.search(r"[.!?]$", text.strip()):
        return True
    if len(text.split()) < 6:
        return True
    return False


def strip_summary_wrapper(text: str) -> str:
    """The model sometimes prefixes a heading, wraps the reply in quotes,
    or leaves stray markdown bold markers around it; clean all of that."""
    text = text.strip()
    text = re.sub(r"^(summary|current state)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip("* \n")
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def write_page_with_summary(path: Path, title: str, summary: str, bullets: list[str]) -> None:
    lines = [f"# {title}", "", "## Summary", "", summary, "", "## Facts"]
    lines.extend(bullets)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Wiki writes (all deterministic; the model never writes files)
# --------------------------------------------------------------------------

def ensure_page(path: Path, title: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n## Facts\n", encoding="utf-8")
    return True


def bullet_core(bullet: str) -> str:
    """A normalized comparison key: bullet minus citation, case, punctuation."""
    core = bullet.strip()
    if core.startswith("- "):
        core = core[2:]
    core = re.sub(r"\s*\(\[\[[^\]]*\]\]\)\s*$", "", core)
    core = re.sub(r"[^a-z0-9 ]+", "", core.lower())
    return re.sub(r"\s+", " ", core).strip()


def bullet_date(bullet: str) -> str:
    """The YYYY-MM-DD citation on a fact bullet, or "" if it has none."""
    match = re.search(r"\(\[\[(\d{4}-\d{2}-\d{2})\]\]\)\s*$", bullet.strip())
    return match.group(1) if match else ""


def sort_bullets_by_date(bullets: list[str]) -> list[str]:
    """Newest journal date first. append_bullet inserts at the top of the
    file in INGESTION order, which only matches journal-date order if notes
    are always ingested chronologically. Backfilling an older note (or
    re-running one out of order, as happened while testing this) breaks
    that. Stage 4 re-sorts by the bullet's own citation date every time it
    touches a page, so the recency-weighted summary (and the page itself)
    stay correct regardless of ingestion order."""
    return sorted(bullets, key=bullet_date, reverse=True)


def facts_within_days(bullets: list[str], today: str, days: int) -> list[str]:
    """Bullets dated within the last `days` days of `today` (inclusive).
    Falls back to all bullets if `today` doesn't parse, and to the single
    newest bullet if the window is empty, so the summary is never built
    from nothing just because an entity's only activity is old."""
    try:
        today_date = date.fromisoformat(today)
    except ValueError:
        return bullets
    kept = []
    for bullet in bullets:
        raw = bullet_date(bullet)
        if not raw:
            continue
        try:
            fact_date = date.fromisoformat(raw)
        except ValueError:
            continue
        if (today_date - fact_date).days <= days:
            kept.append(bullet)
    return kept or bullets[:1]


def append_bullet(path: Path, bullet: str) -> bool:
    """Insert a bullet under '## Facts' (newest first). False if a duplicate."""
    lines = path.read_text(encoding="utf-8").splitlines()
    new_core = bullet_core(bullet)
    for line in lines:
        if line.strip().startswith("- ") and bullet_core(line) == new_core:
            return False
    if "## Facts" not in lines:
        lines.extend(["", "## Facts"])
    insert_at = lines.index("## Facts") + 1
    while insert_at < len(lines) and not lines[insert_at].strip():
        insert_at += 1
    lines.insert(insert_at, bullet)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return True


def quarantine(wiki: Path, date: str, entity_name: str, fields: dict, reason: str) -> None:
    path = wiki / "Quarantine.md"
    if not path.exists():
        path.write_text("# Quarantine\n\nFacts that failed validation. "
                        "Review and file (or delete) by hand.\n\n## Facts\n",
                        encoding="utf-8")
    fact = fields.get("fact", "(no fact text)")
    evidence = fields.get("evidence", "")
    append_bullet(
        path,
        f"- [[{date}]] {entity_name} | {reason}: \"{fact}\" (evidence: \"{evidence}\")",
    )


def rebuild_index(wiki: Path) -> None:
    """Regenerate Index.md: every People/Pets page with a one-line hint."""
    lines = ["# JournalOS Wiki Index", ""]

    def page_hint(path: Path) -> str:
        page_lines = path.read_text(encoding="utf-8").splitlines()
        if "## Summary" in page_lines:
            idx = page_lines.index("## Summary") + 1
            while idx < len(page_lines) and not page_lines[idx].strip():
                idx += 1
            if idx < len(page_lines) and page_lines[idx].strip():
                return page_lines[idx].strip()
        for line in page_lines:
            if line.strip().startswith("- "):
                return line.strip()[2:]
        return "(empty)"

    def add_section(label: str, directory: Path) -> None:
        pages = sorted(directory.glob("*.md")) if directory.exists() else []
        if not pages:
            return
        lines.append(f"## {label}")
        for page in pages:
            rel = page.relative_to(wiki).as_posix()
            lines.append(f"- [{page.stem}]({rel}) — {page_hint(page)}")
        lines.append("")

    add_section("People", wiki / "People")
    add_section("Pets", wiki / "Pets")
    (wiki / "Index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_log(wiki: Path, date: str, summary: str) -> None:
    path = wiki / "Log.md"
    if not path.exists():
        path.write_text("# Ingest Log\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## [{date}] ingest | {summary}\n")


def ensure_scaffold(wiki: Path) -> None:
    for directory in ("People", "Pets"):
        (wiki / directory).mkdir(parents=True, exist_ok=True)
    if not (wiki / "Index.md").exists():
        rebuild_index(wiki)


# --------------------------------------------------------------------------
# Per-note driver
# --------------------------------------------------------------------------

def run_note(note: Path, wiki: Path, out_root: Path, settings: Settings,
            enable_log: bool = False) -> dict:
    date = note.stem
    note_text = note.read_text(encoding="utf-8")
    out_dir = out_root / date
    if enable_log:
        out_dir.mkdir(parents=True, exist_ok=True)
    stages = StageLog(out_dir, settings, enabled=enable_log)
    t0 = time.perf_counter()
    log(f"=== {date} ===")

    # Stage 1: roster.
    roster_reply = stages.call("01-roster", roster_prompt(date, note_text),
                               max_tokens=settings.tokens_roster)
    roster = parse_roster(roster_reply)
    log(f"   roster: people={roster['people']} pets={roster['pets']}")

    # Entities and where each one's page lives. Self goes first only for a
    # stable log/display order.
    entities: list[tuple[str, Path, str]] = [("self", wiki / "People" / "Self.md", "Self")]
    for name in roster["people"]:
        entities.append((name, wiki / "People" / f"{name}.md", name))
    for name in roster["pets"]:
        entities.append((name, wiki / "Pets" / f"{name}.md", name))
    page_by_name = {name: page for name, page, _ in entities}
    title_by_name = {name: title for name, _, title in entities}

    written = 0
    duplicates = 0
    quarantined = 0
    pages_created: list[str] = []
    touched_entities: set[str] = set()

    # Stage 2: one call per paragraph, covering all entities at once. Each
    # call only ever sees that one paragraph, so a fact can't be attributed
    # to someone who isn't mentioned anywhere in it.
    paragraphs = split_paragraphs(note_text)
    for chunk_index, passage in enumerate(paragraphs, 1):
        reply = stages.call(
            f"02-facts-{chunk_index:02d}",
            chunk_facts_prompt(date, roster, passage),
            max_tokens=settings.tokens_facts,
        )
        for fields in parse_labelled_blocks(reply, ("FACT",)):
            fact_text, subject_name, reason = validate_chunk_fact(fields, roster, passage)
            if not fact_text:
                quarantine(wiki, date, fields.get("subject", "unknown"), fields, reason)
                quarantined += 1
                continue
            page = page_by_name[subject_name]
            if ensure_page(page, title_by_name[subject_name]):
                pages_created.append(page.relative_to(wiki).as_posix())
            bullet = f"- {fact_text} ([[{date}]])"
            if append_bullet(page, bullet):
                written += 1
                touched_entities.add(subject_name)
            else:
                duplicates += 1

    # Stage 3: sanity-check every NEW entity page created this run (never
    # "self", which is trivially always valid). See the comment above
    # entity_sanity_prompt for why this checks pages, not every fact.
    for rel_path in list(pages_created):
        page = wiki / rel_path
        if page.stem == "Self":
            continue
        bullets = [line.strip()[2:] for line in page.read_text(encoding="utf-8").splitlines()
                  if line.strip().startswith("- ")]
        if not bullets:
            continue
        reply = stages.call(
            f"03-sanity-{slug(page.stem)}",
            entity_sanity_prompt(page.stem, bullets),
            max_tokens=settings.tokens_sanity,
        )
        is_living, reason = parse_entity_sanity(reply)
        if is_living:
            continue
        for bullet in bullets:
            quarantine(wiki, date, page.stem, {"fact": bullet, "evidence": ""},
                      f"sanity: not a living being ({reason or 'no reason given'})")
        quarantined += len(bullets)
        written -= len(bullets)
        page.unlink()
        pages_created.remove(rel_path)
        touched_entities.discard(page.stem)

    # Stage 4: recency-weighted summary for every entity that got new facts
    # this note (a page nothing changed on today keeps its existing summary).
    for name in sorted(touched_entities):
        page = page_by_name[name]
        if not page.exists():
            continue
        bullets = [line for line in page.read_text(encoding="utf-8").splitlines()
                  if line.strip().startswith("- ")]
        if not bullets:
            continue
        bullets = sort_bullets_by_date(bullets)
        bullet_texts = [line.strip()[2:] for line in bullets]
        if name != "self":
            bullet_texts = [humanize_self_references(text) for text in bullet_texts]
        reply = stages.call(
            f"04-summary-{slug(name)}",
            summary_prompt(name, date, bullet_texts,
                           reasoning=settings.summary_use_reasoning,
                           window_days=settings.summary_window_days),
            max_tokens=settings.tokens_summary,
        )
        summary_text = extract_final_summary(reply)
        if looks_like_unfinished_summary(summary_text):
            # Reasoning ran out of token budget before reaching an answer,
            # or the model never produced a clean final paragraph. Fall
            # back to the plain, non-reasoning prompt rather than writing
            # a truncated reasoning fragment to the page.
            fallback_reply = stages.call(
                f"04-summary-{slug(name)}-fallback",
                summary_prompt(name, date, bullet_texts, reasoning=False,
                               window_days=settings.summary_window_days),
                max_tokens=settings.tokens_summary_fallback,
            )
            summary_text = extract_final_summary(fallback_reply)
        write_page_with_summary(page, title_by_name[name], summary_text, bullets)

    rebuild_index(wiki)
    seconds = round(time.perf_counter() - t0, 1)
    summary = (f"{note.name} | entities: {len(entities)} | facts: {written} written, "
               f"{duplicates} duplicate, {quarantined} quarantined | "
               f"new pages: {len(pages_created)}")
    append_log(wiki, date, summary)

    result = {
        "date": date,
        "entities": [name for name, _, _ in entities],
        "facts_written": written,
        "duplicates_skipped": duplicates,
        "quarantined": quarantined,
        "pages_created": sorted(pages_created),
        "seconds": seconds,
    }
    if enable_log:
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    log(f"   summary: {summary} ({seconds}s)")
    return result
