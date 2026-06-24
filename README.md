# Judge-CS: Code-Switching Invariance for LLM-as-a-Judge

This project builds a lightweight meta-evaluation benchmark from LLMBar and
tests whether an API-based LLM judge changes its preference when the same
instruction-following comparison is presented in English, Chinese, or
Chinese-English code-switched form.

## Layout

- `scripts/judge_cs.py`: end-to-end data preparation, API calls, analysis,
  tables, and figures.
- `data/raw/LLMBar`: cloned upstream LLMBar dataset.
- `data/processed`: sampled items and generated language variants.
- `results`: API judgments, metric summaries, usage counts, and figures.
- `paper`: LaTeX paper source and compiled PDF.

## API Configuration

The API key is intentionally not stored in this repository. Set it only for the
current shell session:

```powershell
$env:JUDGE_API_KEY = "YOUR_KEY"
$env:JUDGE_API_BASE = "https://YOUR-PROXY-OR-PROVIDER/v1/chat/completions"
$env:PYTHONNOUSERSITE = "1"
```

Do not commit a real API key or private proxy endpoint. Keep them in shell
environment variables or a local `.env` file ignored by git.

The example API request used `gpt-4`, but the proxy reported no active channel
for that exact model name during setup. The project therefore uses
`gpt-4.1-mini` for variant generation and evaluates the following judges:
`gpt-4.1-mini`, `claude-haiku-4-5-20251001`, `gemini-2.5-flash`, and
`deepseek-v4-flash`.

## Quick Pilot

```powershell
$env:PYTHONNOUSERSITE = "1"
.\.conda\python.exe scripts\judge_cs.py prepare --limit 30
.\.conda\python.exe scripts\judge_cs.py variants --limit 30 --model gpt-4.1-mini
.\.conda\python.exe scripts\judge_cs.py judge --models gpt-4.1-mini
.\.conda\python.exe scripts\judge_cs.py analyze
```

To reduce API usage while debugging, add `--limit 5` to `variants` and `judge`.

## Full Run

```powershell
$env:PYTHONNOUSERSITE = "1"
.\.conda\python.exe scripts\judge_cs.py prepare --limit 419
.\.conda\python.exe scripts\judge_cs.py variants --model gpt-4.1-mini
.\.conda\python.exe scripts\judge_cs.py judge --models gpt-4.1-mini,claude-haiku-4-5-20251001,gemini-2.5-flash,deepseek-v4-flash --max-tokens 2500 --workers 4
.\.conda\python.exe scripts\judge_cs.py analyze
```

`deepseek-v4-flash` may spend many output tokens in `reasoning_content` before
emitting the final JSON answer. Use `--max-tokens 2500` when including this
model, or replace it with `deepseek-chat` for cleaner non-reasoning output if
that model is available on the endpoint.

## Conda Environment

This project includes a local conda environment at `.conda`.

```powershell
$env:PYTHONNOUSERSITE = "1"
.\.conda\python.exe scripts\judge_cs.py --help
```

On this machine, `PYTHONNOUSERSITE=1` is important because the global user
site-packages directory contains an incompatible pandas/NumPy stack.

## Paper Outputs

The final paper is in `paper/main.pdf`. It is a two-column English paper with
expanded experiments, seven generated figures, and additional aggregate tables.
