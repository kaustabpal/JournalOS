# JournalOS

JournalOS reads your daily journal and turns it into an organized picture of your life, automatically.

The idea is simple: just write. Don't worry about organizing your thoughts, tagging them, or filing them anywhere. Write your day the way you'd talk to a friend, in a single daily journal file. JournalOS reads it and does the organizing for you.

If you journal every day, this adds up to something powerful: a way to actually see how your life is changing over time. How is a relationship with someone growing or fading? How is a project moving forward, or stalling? Journaling already helps you reflect. JournalOS makes that reflection easier to see, by turning scattered daily entries into one connected, evolving picture.

**Right now**, JournalOS can pick out the people, pets, and projects in your life (a side project, an app, a tool, an AI assistant you use) and keep a running page for each one, built from what you've written about them, so you can track not just who is in your life, but what you are working on and how it is progressing.

**Coming next**, it will do the same for problems and trackers: recurring struggles or goals, and things like weight or workout logs, tracked the same way.

Further out, a separate feature: connecting the dots across pages, for example noticing how a project's progress lines up with your mood, or how a habit tracker relates to a relationship, instead of treating each page in isolation.

It runs entirely on your own computer, using a small local AI model. Nothing you write is ever sent anywhere.

To be upfront: tools like Claude or ChatGPT would likely do this job better and faster today. If you do not mind sharing your journal with a cloud AI service, those are a fine choice. JournalOS exists for people who do mind, who want this kind of help without their private thoughts ever leaving their machine. The long-term goal is to close that gap: to get a small, local, purpose-built model to a level that rivals the big cloud models, just for this one job.

Running locally on a small model has a real cost in speed. On the tested setup (a MacBook Air M2 with 16 GB of RAM), ingesting a single day's note takes about 1 to 5 minutes, depending on how long the note is and how many people or pets it mentions. That is the tradeoff: slower than a cloud model, but nothing ever leaves your machine.

Each day's entry should be its own file, named `YYYY-MM-DD.md`. By default JournalOS looks for these files in the `journal/` directory, though you can change that path in `config.yaml`.

If you use Obsidian, its daily notes feature is a natural fit: it opens the same file automatically every day, so you can just write. There is no need to split your thoughts across separate notes; JournalOS handles that separation for you when it ingests the entry.

## Tested setup

The harness is built and tuned specifically for **Gemma 4 E4B**. The exact model and quantization used is:

```text
unsloth/gemma-4-E4B-it-UD-MLX-4bit
```

Every prompt, token budget, and validation step is calibrated for this small 4-bit model; a different model may need the prompts retuned. It has been tested on a **MacBook Air M2 with 16 GB of RAM**, where this quant runs comfortably.

## Installation

**Requirements**

- Conda or Miniconda
- Python 3.10+
- An OpenAI-compatible chat completions server to point `ingest` at (see below)

**Steps**

```bash
git clone https://github.com/kaustabpal/JournalOS.git
cd JournalOS
conda create -n journalos python=3.10 -y
conda activate journalos
pip install -r requirements.txt
```

`ingest` itself is nearly dependency-free: its only requirement is PyYAML, used to read `config.yaml`. It works on any OS.

**If you're on macOS with Apple Silicon**, JournalOS bundles a ready-to-run server (`./serve`, using `mlx-vlm`). Install its dependencies:

```bash
pip install -r requirements-serve.txt
```

This installs `mlx-vlm` from the JournalOS-compatible fork at `kaustabpal/mlx-vlm`, required for Gemma 4 support and a server compatibility patch. `mlx-vlm` only runs on Apple Silicon.

**On Linux or Windows**, `./serve` won't work, since it depends on Apple's MLX framework. `ingest` still will: run any OpenAI-compatible server yourself (e.g. `vllm`, `llama.cpp`'s server, `text-generation-webui`, Ollama's OpenAI-compatible endpoint) and point `ingest` at it by setting `model.base_url` and `model.name` in `config.yaml`.

## Usage

### 1. Start a model server

**macOS (Apple Silicon)**, using the bundled server:

```bash
./serve
```

Leave this running in its own terminal. By default it serves `unsloth/gemma-4-E4B-it-UD-MLX-4bit` at `http://127.0.0.1:8090`, with non-strict parameter loading (needed for this checkpoint). `./serve` reads all of this from `config.yaml`, so to use a different model, port, host, or strict loading, edit the `model.name` and `serve:` settings there (see [Configuration](#configuration)), then run `./serve` again.

**Linux, Windows, or any other server you prefer**: run your own OpenAI-compatible chat completions server and leave it running. `ingest` doesn't care what's behind the endpoint, only that it speaks the same API `./serve` does. Point `ingest` at it by editing `model.base_url` (and `model.name`) in `config.yaml`.

### 2. Add journal notes

In a second terminal, copy one Markdown file per day into `journal/`, named `YYYY-MM-DD.md`:

```bash
cp ~/Notes/Journal/2026-06-02.md journal/
```

`journal/` ships with five synthetic example entries (`2026-01-01.md` through `2026-01-05.md`) so you can try `python ingest` immediately after cloning and see what the wiki output looks like, without needing your own journal yet. Delete them before adding your own notes:

```bash
rm journal/2026-01-0*.md
```

`journal/` is git-ignored, so your notes are never committed, and JournalOS never writes to files in this folder.

### 3. Run ingest

```bash
conda activate journalos
python ingest
```

That's the command you'll use day to day: no arguments, run it whenever you've added or edited notes. It figures out what's new on its own. By default you'll see a single self-updating progress bar per note (roster, facts, sanity check, summary, each 25% of the bar), so you always know how far along a note is without a wall of text.

### Arguments

| Command | When to use it |
|---|---|
| `python ingest` | Everyday use. Processes only notes that are new or edited since the last run you ran this way. |
| `python ingest 2026-06-02` | Ingest one specific note by date (also accepts a bare filename like `2026-06-02.md`). |
| `python ingest 2026-06-01..2026-06-10` | Ingest a specific date range, inclusive. |
| `--reset-wiki` | Wipes `wiki/` first, then rebuilds it. Use this after changing how the pipeline extracts facts, or if the wiki looks wrong and you want a clean rebuild. With no notes argument, it rebuilds from every note in `journal/`. Combined with a specific note or range, it still wipes the *whole* wiki but only rebuilds from those notes, so facts from every other note are permanently lost; use it plain (no notes argument) unless that is exactly what you want. |
| `--log` | Add to any of the above to record every stage's prompt/output under `logs/`. Off by default. Turn it on when a run fails or produces something suspicious, then re-run just that note with `--log` to see exactly where it went wrong. Writes files; doesn't print anything extra to the terminal. |
| `--verbose` / `-v` | Replace the default progress bar with a live narration of each stage and what the model actually replied, printed to the terminal as the run happens. Use this to watch the pipeline reason through a note in real time, the way you'd watch a frontier model think. Independent of `--log`, and off by default since it's a lot of text for everyday use. |
| `--config <path>` | Use a config file other than `./config.yaml`. Rarely needed; see [Configuration](#configuration). |

### Output

```text
wiki/
  People/         one page per person (People/Self.md is you, the journal author)
  Pets/           one page per pet
  Projects/       one page per project, app, tool, or AI assistant
  Index.md        regenerated catalog of every page
  Log.md          append-only ingest log
  Quarantine.md   facts/pages that failed validation, for human review
```

Each page has a `## Summary` (recency-weighted prose) followed by `## Facts` (dated bullets citing the source note, e.g. `([[2026-06-02]])`). To have those citations resolve as links in Obsidian, open this repository's root as the vault (not just `wiki/`), so `journal/YYYY-MM-DD.md` is reachable too.

## How it works

This is a local-first pipeline designed for small models: the harness (Python) makes every structural decision, and the model only answers small, closed questions. Currently scoped to people, pets, and projects (no topic/problem pages yet). Each note goes through four stages:

1. **Roster**: one call listing the people, pets, and projects (apps, tools, AI assistants) mentioned in the note.
2. **Facts**: one call per paragraph, covering every known entity (self and everyone in the roster) at once. Chunking by paragraph means a fact can never be attributed to someone who isn't mentioned anywhere in that paragraph. Evidence quotes and subjects are validated in Python.
3. **Sanity check**: one call per newly-created page, asking which of person/pet/project best fits the name, or whether it's none of those (a company, a place) and shouldn't have a page at all. If the roster put it in the wrong category, the page is moved to the right one instead of being discarded. Only runs on new entities, not every fact.
4. **Summary**: one call per entity that got new facts this note, writing a short recency-weighted prose summary at the top of their page. Falls back to a plain non-reasoning prompt if the reasoning attempt runs out of its token budget without reaching an answer.

Facts and pages that fail validation at any stage go to `wiki/Quarantine.md` for human review instead of being guessed at. Re-running a note is always safe: bullets are deduplicated, so nothing is filed twice.

### How `python ingest` (no args) decides what's new

It hashes each note's content and compares it against a local record, `.journalos-last-ingest` (also git-ignored, never committed). A note is processed if it's new or its hash has changed since the last no-args run, so editing an old note re-triggers ingestion of just that note. The record only advances once every note in a run succeeds, so a failed note is retried on the next run instead of being silently skipped.

## Repository Layout

```text
JournalOS/
  serve                  # starts the bundled local inference server (macOS/Apple Silicon only)
  ingest                 # command users run (cross-platform)
  config.yaml            # every setting for ingest and ./serve; edit this
  requirements.txt       # deps for ingest, just PyYAML
  requirements-serve.txt # deps for ./serve: mlx-vlm and friends (macOS only)
  journal/               # your daily notes (git-ignored)
  wiki/                  # generated wiki output
  logs/                  # only exists if you pass --log
  src/                   # ingest implementation (wiki_ingest.py)
  .journalos-last-ingest # local record of what's been ingested (git-ignored)
```

## Configuration

All settings live in [`config.yaml`](config.yaml). Both `ingest` and
`./serve` read it directly; there are no environment variables to set.
Just edit it: it controls the paths, the model, the server host/port,
per-stage token budgets, and the summary settings. Every value in the
file is also its default, and anything you delete falls back to that
default, so you only need to keep the lines you actually change. Each
option is documented in the file's comments.

`model.name` is shared by both tools, so set it once and `./serve` and
`ingest` stay in sync. `model.base_url` (used by `ingest`) must point at
the same host and port as the `serve:` section (used by `./serve`), since
that is how `ingest` reaches the server `./serve` starts.

Use `--config <path>` to have `ingest` read a config file other than
`./config.yaml`.

## License

GPL-3.0. See [LICENSE](LICENSE).
