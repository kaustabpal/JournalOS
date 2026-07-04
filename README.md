# JournalOS

JournalOS turns daily journal notes into a structured personal wiki using a local LLM.

Copy Markdown journal notes into `journal/`, run `python ingest <date>`, and JournalOS writes connected wiki pages into `wiki/`. Every extracted fact keeps a dated source citation.

## Status

This is an early local-first ingest pipeline. It is designed to guide the model with structure, not force every decision with rules. The current frozen pipeline is:

1. Build note context.
2. Extract plain-text facts.
3. Place facts into wiki pages.
4. Repair incomplete placements.
5. Write the wiki pages and logs.

## Repository Layout

```text
JournalOS/
  ingest                 # command users run
  journal/               # copy daily notes here
  wiki/                  # generated wiki output
  logs/                  # stage outputs and diagnostics
  src/journalos/         # ingest implementation
```

## Requirements

- Python 3.10+
- A running OpenAI-compatible local chat-completions server
- A model that can follow structured prompts

The tested setup uses `mlx_vlm.server` with:

```text
unsloth/gemma-4-E4B-it-UD-MLX-4bit
```

The ingest script expects the server at:

```text
http://127.0.0.1:8090/v1
```

You can change `MODEL` and `BASE_URL` at the top of `ingest`.

## Quick Start

1. Clone the repo.

```bash
git clone <your-repo-url> JournalOS
cd JournalOS
```

2. Start your local LLM server.

It must expose an OpenAI-compatible endpoint at:

```text
http://127.0.0.1:8090/v1/chat/completions
```

3. Copy notes into `journal/`.

Use one Markdown file per day:

```text
journal/2026-06-02.md
journal/2026-06-03.md
```

4. Ingest one note.

```bash
python ingest 2026-06-02
```

5. Ingest a date range.

```bash
python ingest 2026-06-01..2026-06-10
```

6. Review the output.

```text
wiki/   generated wiki pages
logs/   prompts, model outputs, repair logs, and report.json
```

## Notes

- `--reset-wiki` clears the generated `wiki/` folder before running.
- Quote mismatch warnings are treated as diagnostics, not hard failures.
- The pipeline writes a primary page first, then profile side effects.
- Person profiles are generated under `wiki/Wiki/social-connections/profiles/`.

## Example

```bash
cp ~/Notes/Journal/2026-06-02.md journal/
python ingest 2026-06-02 --reset-wiki
```

After the run:

```text
wiki/Wiki/
logs/2026-06-02/
logs/report.json
```

## License

Add a license before publishing.
