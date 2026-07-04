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
  serve                  # starts the local inference server
  ingest                 # command users run
  journal/               # copy daily notes here
  wiki/                  # generated wiki output
  logs/                  # stage outputs and diagnostics
  src/journalos/         # ingest implementation
```

## Requirements

- macOS with Apple Silicon
- Conda or Miniconda
- Python 3.10+
- Enough free memory to run the local model

The tested setup uses `mlx_vlm.server` with:

```text
unsloth/gemma-4-E4B-it-UD-MLX-4bit
```

The current upstream `mlx-vlm` repo has a dedicated Gemma 4 backend and documents `google/gemma-4-e2b-it`, `google/gemma-4-e4b-it`, `google/gemma-4-26b-a4b-it`, and `google/gemma-4-31b-it`.

The ingest script expects the server at:

```text
http://127.0.0.1:8090/v1
```

You can change `MODEL` and `BASE_URL` at the top of `ingest`. You can also set server environment variables before running `./serve`.

## Quick Start

1. Clone the repo.

```bash
git clone <your-repo-url> JournalOS
cd JournalOS
```

2. Create a conda environment.

```bash
conda create -n journalos python=3.10 -y
conda activate journalos
```

3. Install requirements.

```bash
pip install -r requirements.txt
```

JournalOS installs `mlx-vlm` from the upstream GitHub `main` branch so Gemma 4 support is available even if the PyPI package lags behind the repository.

4. Start the local inference server.

```bash
./serve
```

By default this starts:

```text
model: unsloth/gemma-4-E4B-it-UD-MLX-4bit
host: 127.0.0.1
port: 8090
prefill step size: 2048
```

To use another model or port:

```bash
JOURNALOS_MODEL=/path/to/local/model JOURNALOS_PORT=8090 ./serve
```

It must expose an OpenAI-compatible endpoint at:

```text
http://127.0.0.1:8090/v1/chat/completions
```

5. Copy notes into `journal/`.

Use one Markdown file per day:

```text
journal/2026-06-02.md
journal/2026-06-03.md
```

6. Ingest one note in a second terminal.

```bash
conda activate journalos
python ingest 2026-06-02
```

7. Ingest a date range.

```bash
python ingest 2026-06-01..2026-06-10
```

8. Review the output.

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
