"""Judge-LS experiment runner.

This script prepares a small LLMBar-derived benchmark, generates Chinese and
language-switched variants, calls an OpenAI-compatible chat-completions endpoint as
an LLM judge, and summarizes language-invariance metrics.

The API key is read from JUDGE_API_KEY. Do not hard-code secrets in this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "LLMBar" / "Dataset" / "LLMBar"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PAPER = ROOT / "paper"

DEFAULT_API_BASE = "http://35.220.164.252:3888/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"

QUALITY_SYSTEM = (
    "You are a careful evaluator of instruction-following responses. "
    "Judge only which response better follows the given instruction. "
    "Do not favor a response because of its language, length, politeness, or formatting "
    "unless that directly affects correctness or instruction following. "
    "Return strict JSON with keys winner, confidence, and reason. "
    "winner must be exactly one of A, B, Tie."
)

VARIANT_SYSTEM = (
    "You generate label-preserving language variants for evaluation research. "
    "Preserve the relative quality of the two candidate outputs. "
    "Do not correct factual, reasoning, or instruction-following errors."
)

TIE_SYSTEM = (
    "You are evaluating whether two responses are equally useful. "
    "If two responses are semantically equivalent and equally helpful, choose Tie. "
    "Do not prefer English or Chinese merely because of language. "
    "Return strict JSON with keys winner, confidence, and reason. "
    "winner must be exactly one of A, B, Tie."
)


# Prices are for rough what-if estimates only. The provided proxy endpoint may
# bill differently, so the script always reports token counts separately.
PRICE_USD_PER_M = {
    "gpt-4": (30.0, 60.0),  # legacy public GPT-4 reference price
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-5.4-mini": (0.75, 4.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "deepseek-v4-flash": (0.14, 0.28),
}


def ensure_dirs() -> None:
    for path in (PROCESSED, RESULTS, FIGURES, PAPER):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: str) -> str:
    h = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return h


def approx_tokens(text: str) -> int:
    # A conservative language-agnostic approximation for budget reporting.
    return max(1, math.ceil(len(text) / 3.6))


def load_llmbar_items() -> List[Dict[str, Any]]:
    sources = [
        ("Natural", RAW_ROOT / "Natural" / "dataset.json"),
        ("Adv-Neighbor", RAW_ROOT / "Adversarial" / "Neighbor" / "dataset.json"),
        ("Adv-GPTInst", RAW_ROOT / "Adversarial" / "GPTInst" / "dataset.json"),
        ("Adv-GPTOut", RAW_ROOT / "Adversarial" / "GPTOut" / "dataset.json"),
        ("Adv-Manual", RAW_ROOT / "Adversarial" / "Manual" / "dataset.json"),
    ]
    items: List[Dict[str, Any]] = []
    for source, path in sources:
        if not path.exists():
            continue
        data = read_json(path)
        for idx, row in enumerate(data):
            item_id = stable_id(source, str(idx), row["input"], row["output_1"], row["output_2"])
            items.append(
                {
                    "id": item_id,
                    "source": source,
                    "source_index": idx,
                    "input": row["input"],
                    "output_1": row["output_1"],
                    "output_2": row["output_2"],
                    "label": int(row["label"]),
                }
            )
    if not items:
        raise FileNotFoundError(f"No LLMBar data found under {RAW_ROOT}")
    return items


def cmd_prepare(args: argparse.Namespace) -> None:
    ensure_dirs()
    rng = random.Random(args.seed)
    items = load_llmbar_items()
    if args.limit >= len(items):
        sampled = list(items)
        rng.shuffle(sampled)
        sources = sorted({item["source"] for item in sampled})
        out = PROCESSED / "sample_items.jsonl"
        write_jsonl(out, sampled)
        write_json(
            PROCESSED / "sample_metadata.json",
            {
                "seed": args.seed,
                "limit": args.limit,
                "num_items": len(sampled),
                "sampling": "all_items_shuffled",
                "sources": {s: sum(1 for x in sampled if x["source"] == s) for s in sources},
            },
        )
        print(f"Wrote all {len(sampled)} LLMBar items to {out}")
        return
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_source.setdefault(item["source"], []).append(item)
    sampled: List[Dict[str, Any]] = []
    sources = sorted(by_source)
    per_source = max(1, args.limit // len(sources))
    for source in sources:
        bucket = list(by_source[source])
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_source])
    if len(sampled) < args.limit:
        chosen = {item["id"] for item in sampled}
        rest = [item for item in items if item["id"] not in chosen]
        rng.shuffle(rest)
        sampled.extend(rest[: args.limit - len(sampled)])
    sampled = sampled[: args.limit]
    rng.shuffle(sampled)
    out = PROCESSED / "sample_items.jsonl"
    write_jsonl(out, sampled)
    write_json(
        PROCESSED / "sample_metadata.json",
        {
            "seed": args.seed,
            "limit": args.limit,
            "num_items": len(sampled),
            "sources": {s: sum(1 for x in sampled if x["source"] == s) for s in sources},
        },
    )
    print(f"Wrote {len(sampled)} sampled items to {out}")


@dataclass
class ApiClient:
    api_base: str
    api_key: str
    timeout: int = 120
    retries: int = 3

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error: Optional[str] = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(self.api_base, headers=headers, json=body, timeout=self.timeout)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    time.sleep(2 ** attempt)
                    continue
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return {
                    "content": content,
                    "raw": data,
                    "usage": data.get("usage", {}),
                }
            except Exception as exc:  # noqa: BLE001 - preserve API failure text
                last_error = repr(exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API call failed after {self.retries} attempts: {last_error}")


def get_client(args: argparse.Namespace) -> ApiClient:
    api_key = os.environ.get("JUDGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JUDGE_API_KEY is not set. Set it in the shell before calling the API.")
    api_base = os.environ.get("JUDGE_API_BASE", args.api_base or DEFAULT_API_BASE).strip()
    return ApiClient(api_base=api_base, api_key=api_key, timeout=args.timeout, retries=args.retries)


def extract_json_obj(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_variant_prompt(item: Dict[str, Any]) -> str:
    return f"""
Create two label-preserving variants of this comparison item.

Requirements:
- zh: translate the instruction and both candidate outputs into natural Simplified Chinese.
- cs: create a natural Chinese-English language-switched version. Keep important task terms or named concepts in English where useful, but make the item readable.
- Preserve every factual, mathematical, formatting, and instruction-following mistake. Do not improve either candidate.
- Keep output_1 aligned with output_1 and output_2 aligned with output_2.
- Return strict JSON only, with this schema:
{{
  "zh": {{"input": "...", "output_1": "...", "output_2": "..."}},
  "cs": {{"input": "...", "output_1": "...", "output_2": "..."}}
}}

Original item:
Instruction:
{item["input"]}

Output 1:
{item["output_1"]}

Output 2:
{item["output_2"]}
""".strip()


def cmd_variants(args: argparse.Namespace) -> None:
    ensure_dirs()
    sample_path = PROCESSED / "sample_items.jsonl"
    items = list(iter_jsonl(sample_path))
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise RuntimeError(f"No sampled items found at {sample_path}; run prepare first.")

    client = get_client(args)
    out = PROCESSED / "variants.jsonl"
    done = {row["id"] for row in iter_jsonl(out) if row.get("status") == "ok"} if out.exists() and args.resume else set()
    if not args.resume and out.exists():
        out.unlink()

    pending = [(i, item) for i, item in enumerate(items, start=1) if item["id"] not in done]
    print(f"Variant generation pending={len(pending)} skipped={len(items)-len(pending)} workers={args.workers}")

    def process_variant(i: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        messages = [
            {"role": "system", "content": VARIANT_SYSTEM},
            {"role": "user", "content": generate_variant_prompt(item)},
        ]
        prompt_text = json.dumps(messages, ensure_ascii=False)
        row: Dict[str, Any] = {
            "id": item["id"],
            "source": item["source"],
            "label": item["label"],
            "en": {
                "input": item["input"],
                "output_1": item["output_1"],
                "output_2": item["output_2"],
            },
            "variant_model": args.model,
            "prompt_tokens_approx": approx_tokens(prompt_text),
        }
        try:
            result = client.chat(args.model, messages, temperature=0.0, max_tokens=args.max_tokens)
            parsed = extract_json_obj(result["content"])
            for lang in ("zh", "cs"):
                if lang not in parsed:
                    raise ValueError(f"missing {lang} in variant response")
                row[lang] = {
                    "input": str(parsed[lang]["input"]),
                    "output_1": str(parsed[lang]["output_1"]),
                    "output_2": str(parsed[lang]["output_2"]),
                }
            row["usage"] = result.get("usage", {})
            row["completion_tokens_approx"] = approx_tokens(result["content"])
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = repr(exc)
        time.sleep(args.sleep)
        return i, row

    if args.workers <= 1:
        for i, item in pending:
            idx, row = process_variant(i, item)
            append_jsonl(out, row)
            print(f"[{idx}/{len(items)}] variants {row['id']} -> {row['status']}")
    else:
        write_lock = Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_variant, i, item) for i, item in pending]
            completed = 0
            for future in as_completed(futures):
                idx, row = future.result()
                completed += 1
                with write_lock:
                    append_jsonl(out, row)
                    print(f"[{completed}/{len(pending)}] variants {row['id']} original_index={idx} -> {row['status']}")


def quality_prompt(variant: Dict[str, str], a: str, b: str, language: str) -> str:
    return f"""
Language condition: {language}

Instruction:
{variant["input"]}

Response A:
{a}

Response B:
{b}

Which response better follows the instruction?
Return strict JSON only:
{{"winner":"A|B|Tie","confidence":0.0,"reason":"brief reason"}}
""".strip()


def tie_probe_prompt(instruction: str, a: str, b: str) -> str:
    return f"""
The two responses below are intended to be translation-equivalent answers to the same instruction.
Choose Tie if they are semantically equivalent and equally useful.

Instruction:
{instruction}

Response A:
{a}

Response B:
{b}

Return strict JSON only:
{{"winner":"A|B|Tie","confidence":0.0,"reason":"brief reason"}}
""".strip()


def normalize_winner(raw: Any) -> str:
    text = str(raw).strip().upper()
    if text in {"A", "RESPONSE A", "OUTPUT A"}:
        return "A"
    if text in {"B", "RESPONSE B", "OUTPUT B"}:
        return "B"
    if "TIE" in text or "DRAW" in text or text in {"NONE", "EQUAL"}:
        return "Tie"
    return "Invalid"


def parse_judgment(content: str) -> Dict[str, Any]:
    try:
        parsed = extract_json_obj(content)
    except Exception:
        parsed = {}
        winner_match = re.search(
            r'"?winner"?\s*[:=]\s*"?\b(A|B|Tie)\b"?|(?:^|\b)(A|B|Tie)(?:\b|$)',
            content,
            flags=re.IGNORECASE,
        )
        if winner_match:
            parsed["winner"] = next(g for g in winner_match.groups() if g)
        conf_match = re.search(r'"?confidence"?\s*[:=]\s*([01](?:\.\d+)?)', content, flags=re.IGNORECASE)
        if conf_match:
            parsed["confidence"] = conf_match.group(1)
        parsed["reason"] = content[:1000]
    winner = normalize_winner(parsed.get("winner", "Invalid"))
    confidence_raw = parsed.get("confidence", None)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = None
    return {
        "winner": winner,
        "confidence": confidence,
        "reason": str(parsed.get("reason", ""))[:1000],
    }


def build_quality_tasks(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for lang in ("en", "zh", "cs"):
        if lang not in row:
            continue
        variant = row[lang]
        tasks.append(
            {
                "task_type": "quality",
                "item_id": row["id"],
                "source": row["source"],
                "language": lang,
                "order": "orig",
                "gold_label": row["label"],
                "prompt": quality_prompt(variant, variant["output_1"], variant["output_2"], lang),
            }
        )
        tasks.append(
            {
                "task_type": "quality",
                "item_id": row["id"],
                "source": row["source"],
                "language": lang,
                "order": "swap",
                "gold_label": row["label"],
                "prompt": quality_prompt(variant, variant["output_2"], variant["output_1"], lang),
            }
        )
    return tasks


def build_tie_tasks(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "zh" not in row:
        return []
    label = int(row["label"])
    out_key = f"output_{label}"
    en = row["en"][out_key]
    zh = row["zh"][out_key]
    instruction = row["zh"]["input"]
    return [
        {
            "task_type": "tie_probe",
            "item_id": row["id"],
            "source": row["source"],
            "language": "en_vs_zh",
            "order": "en_zh",
            "gold_label": "Tie",
            "prompt": tie_probe_prompt(instruction, en, zh),
        },
        {
            "task_type": "tie_probe",
            "item_id": row["id"],
            "source": row["source"],
            "language": "en_vs_zh",
            "order": "zh_en",
            "gold_label": "Tie",
            "prompt": tie_probe_prompt(instruction, zh, en),
        },
    ]


def task_key(model: str, task: Dict[str, Any]) -> str:
    return "|".join(
        [
            model,
            task["task_type"],
            task["item_id"],
            task["language"],
            task["order"],
        ]
    )


def cmd_judge(args: argparse.Namespace) -> None:
    ensure_dirs()
    rows = [row for row in iter_jsonl(PROCESSED / "variants.jsonl") if row.get("status") == "ok"]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No generated variants found; run variants first.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = get_client(args)
    out = RESULTS / "judgments.jsonl"
    done = {row["request_key"] for row in iter_jsonl(out) if row.get("status") == "ok"} if out.exists() and args.resume else set()
    if not args.resume and out.exists():
        out.unlink()

    tasks: List[Tuple[str, Dict[str, Any]]] = []
    for model in models:
        for row in rows:
            for task in build_quality_tasks(row):
                tasks.append((model, task))
            for task in build_tie_tasks(row):
                tasks.append((model, task))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    pending = [(idx, model, task) for idx, (model, task) in enumerate(tasks, start=1) if task_key(model, task) not in done]
    print(f"Judgment pending={len(pending)} skipped={len(tasks)-len(pending)} workers={args.workers}")

    def process_judgment(idx: int, model: str, task: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        key = task_key(model, task)
        messages = [
            {
                "role": "system",
                "content": TIE_SYSTEM if task["task_type"] == "tie_probe" else QUALITY_SYSTEM,
            },
            {"role": "user", "content": task["prompt"]},
        ]
        prompt_text = json.dumps(messages, ensure_ascii=False)
        record = {
            "request_key": key,
            "model": model,
            "task_type": task["task_type"],
            "item_id": task["item_id"],
            "source": task["source"],
            "language": task["language"],
            "order": task["order"],
            "gold_label": task["gold_label"],
            "prompt_tokens_approx": approx_tokens(prompt_text),
        }
        try:
            result = client.chat(model, messages, temperature=0.0, max_tokens=args.max_tokens)
            record["raw_content"] = result["content"]
            parsed = parse_judgment(result["content"])
            if parsed["winner"] == "Invalid":
                raise ValueError(f"Could not parse winner from response: {result['content'][:500]!r}")
            record.update(parsed)
            record["usage"] = result.get("usage", {})
            record["completion_tokens_approx"] = approx_tokens(result["content"])
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = repr(exc)
        time.sleep(args.sleep)
        return idx, record

    if args.workers <= 1:
        for idx, model, task in pending:
            original_idx, record = process_judgment(idx, model, task)
            append_jsonl(out, record)
            print(
                f"[{original_idx}/{len(tasks)}] judge {record['model']} {record['task_type']} "
                f"{record['item_id']} {record['language']} {record['order']} -> {record['status']}"
            )
    else:
        write_lock = Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_judgment, idx, model, task) for idx, model, task in pending]
            completed = 0
            for future in as_completed(futures):
                original_idx, record = future.result()
                completed += 1
                with write_lock:
                    append_jsonl(out, record)
                    print(
                        f"[{completed}/{len(pending)}] judge {record['model']} {record['task_type']} "
                        f"{record['item_id']} {record['language']} {record['order']} "
                        f"original_index={original_idx} -> {record['status']}"
                    )


def normalized_choice_for_quality(row: Dict[str, Any]) -> str:
    winner = row.get("winner")
    if winner == "Tie":
        return "Tie"
    if winner not in {"A", "B"}:
        return "Invalid"
    if row["order"] == "orig":
        return "1" if winner == "A" else "2"
    return "2" if winner == "A" else "1"


def tie_probe_language_choice(row: Dict[str, Any]) -> str:
    winner = row.get("winner")
    if winner == "Tie":
        return "Tie"
    if winner not in {"A", "B"}:
        return "Invalid"
    if row["order"] == "en_zh":
        return "EN" if winner == "A" else "ZH"
    return "ZH" if winner == "A" else "EN"


def mean(vals: List[float]) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


MODEL_SHORT = {
    "claude-haiku-4-5-20251001": "Claude Haiku",
    "deepseek-v4-flash": "DeepSeek",
    "gemini-2.5-flash": "Gemini Flash",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-mini (variant-generation)": "Variant gen.",
}


def short_model_name(model: str) -> str:
    return MODEL_SHORT.get(model, model)


LANG_DISPLAY = {"en": "EN", "zh": "ZH", "cs": "LS"}
COMPARISON_DISPLAY = {"en_vs_zh": "EN--ZH", "en_vs_cs": "EN--LS"}
LANG_COLORS = {"en": "#4C78A8", "zh": "#F58518", "cs": "#54A24B"}
GROUP_BAR_WIDTH = 0.24
BAR_EDGE = {"edgecolor": "white", "linewidth": 0.6}


def display_lang(lang: Any) -> str:
    return LANG_DISPLAY.get(str(lang).lower(), str(lang).upper())


def display_comparison(comparison: Any) -> str:
    text = str(comparison)
    return COMPARISON_DISPLAY.get(text, text.replace("en_vs_", "EN--").upper())


def apply_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )


def style_bar_axis(ax: Any) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def proportion_ci_wilson(count: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mean_ci_normal(vals: List[float], z: float = 1.96) -> Tuple[float, float]:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    n = len(vals)
    if n <= 1:
        m = vals[0] if vals else float("nan")
        return (m, m)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    margin = z * math.sqrt(var / n)
    return (max(0.0, m - margin), min(1.0, m + margin))


def binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    observed = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        if prob <= observed + 1e-15:
            total += prob
    return min(1.0, total)


def fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if math.isnan(val):
        return ""
    return f"{100 * val:.{digits}f}"


def fmt_ci(low: Any, high: Any, digits: int = 1) -> str:
    return f"[{fmt_pct(low, digits)}, {fmt_pct(high, digits)}]"


def normalized_confidence_value(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val):
        return None
    if 1.0 < val <= 100.0:
        val = val / 100.0
    if val < 0.0 or val > 1.0:
        return None
    return val


def pct(x: float) -> float:
    if math.isnan(x):
        return float("nan")
    return 100.0 * x


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_by(rows: Iterable[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    return groups


def metric_quality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (model, lang), group in sorted(group_by(rows, ("model", "language")).items()):
        valid = [r for r in group if r.get("normalized_choice") in {"1", "2", "Tie"}]
        n = len(valid)
        if not n:
            continue
        strict_count = sum(1 for r in valid if r["normalized_choice"] == str(r["gold_label"]))
        strict = strict_count / n
        half_scores = [
            1.0 if r["normalized_choice"] == str(r["gold_label"]) else 0.5 if r["normalized_choice"] == "Tie" else 0.0
            for r in valid
        ]
        half = sum(half_scores) / n
        a_count = sum(1 for r in valid if r.get("winner") == "A")
        tie_count = sum(1 for r in valid if r.get("winner") == "Tie")
        a_rate = a_count / n
        tie_rate = tie_count / n
        strict_low, strict_high = proportion_ci_wilson(strict_count, n)
        half_low, half_high = mean_ci_normal(half_scores)
        a_low, a_high = proportion_ci_wilson(a_count, n)
        tie_low, tie_high = proportion_ci_wilson(tie_count, n)
        out.append(
            {
                "model": model,
                "model_short": short_model_name(model),
                "language": lang,
                "n": n,
                "accuracy_strict": round(strict, 4),
                "accuracy_strict_ci_low": round(strict_low, 4),
                "accuracy_strict_ci_high": round(strict_high, 4),
                "accuracy_tie_half": round(half, 4),
                "accuracy_tie_half_ci_low": round(half_low, 4),
                "accuracy_tie_half_ci_high": round(half_high, 4),
                "position_A_rate": round(a_rate, 4),
                "position_A_rate_ci_low": round(a_low, 4),
                "position_A_rate_ci_high": round(a_high, 4),
                "tie_rate": round(tie_rate, 4),
                "tie_rate_ci_low": round(tie_low, 4),
                "tie_rate_ci_high": round(tie_high, 4),
                "avg_confidence": round(mean([normalized_confidence_value(r.get("confidence")) for r in valid]), 4),
            }
        )
    return out


def metric_invariance(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    quality = [r for r in rows if r["task_type"] == "quality" and r.get("normalized_choice") in {"1", "2", "Tie"}]
    for (model, target_lang), _ in sorted({(r["model"], r["language"]): True for r in quality if r["language"] in {"zh", "cs"}}.items()):
        pairs = []
        for (item_id, order), group in group_by(
            [r for r in quality if r["model"] == model and r["language"] in {"en", target_lang}],
            ("item_id", "order"),
        ).items():
            by_lang = {r["language"]: r for r in group}
            if "en" in by_lang and target_lang in by_lang:
                pairs.append((by_lang["en"], by_lang[target_lang]))
        if not pairs:
            continue
        flips = sum(1 for en, other in pairs if en["normalized_choice"] != other["normalized_choice"])
        gold_flips = sum(
            1
            for en, other in pairs
            if (en["normalized_choice"] == str(en["gold_label"])) != (other["normalized_choice"] == str(other["gold_label"]))
        )
        flip_low, flip_high = proportion_ci_wilson(flips, len(pairs))
        gold_low, gold_high = proportion_ci_wilson(gold_flips, len(pairs))
        out.append(
            {
                "model": model,
                "model_short": short_model_name(model),
                "comparison": f"en_vs_{target_lang}",
                "n_pairs": len(pairs),
                "language_invariance_flip_rate": round(flips / len(pairs), 4),
                "language_invariance_flip_rate_ci_low": round(flip_low, 4),
                "language_invariance_flip_rate_ci_high": round(flip_high, 4),
                "gold_correctness_flip_rate": round(gold_flips / len(pairs), 4),
                "gold_correctness_flip_rate_ci_low": round(gold_low, 4),
                "gold_correctness_flip_rate_ci_high": round(gold_high, 4),
            }
        )
    return out


def metric_position(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    quality = [r for r in rows if r["task_type"] == "quality" and r.get("normalized_choice") in {"1", "2", "Tie"}]
    for (model, lang), group in sorted(group_by(quality, ("model", "language")).items()):
        pairs = []
        for item_id, item_group in group_by(group, ("item_id",)).items():
            by_order = {r["order"]: r for r in item_group}
            if "orig" in by_order and "swap" in by_order:
                pairs.append((by_order["orig"], by_order["swap"]))
        if not pairs:
            continue
        inconsistent = sum(1 for a, b in pairs if a["normalized_choice"] != b["normalized_choice"])
        low, high = proportion_ci_wilson(inconsistent, len(pairs))
        out.append(
            {
                "model": model,
                "model_short": short_model_name(model),
                "language": lang,
                "n_pairs": len(pairs),
                "position_inconsistency_rate": round(inconsistent / len(pairs), 4),
                "position_inconsistency_rate_ci_low": round(low, 4),
                "position_inconsistency_rate_ci_high": round(high, 4),
            }
        )
    return out


def metric_tie_probe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    probes = [r for r in rows if r["task_type"] == "tie_probe" and r.get("language_choice") in {"EN", "ZH", "Tie"}]
    for model, group in sorted(group_by(probes, ("model",)).items()):
        g = group
        n = len(g)
        if not n:
            continue
        en = sum(1 for r in g if r["language_choice"] == "EN")
        zh = sum(1 for r in g if r["language_choice"] == "ZH")
        tie = sum(1 for r in g if r["language_choice"] == "Tie")
        non_tie = en + zh
        en_low, en_high = proportion_ci_wilson(en, n)
        zh_low, zh_high = proportion_ci_wilson(zh, n)
        tie_low, tie_high = proportion_ci_wilson(tie, n)
        out.append(
            {
                "model": model[0] if isinstance(model, tuple) else model,
                "model_short": short_model_name(model[0] if isinstance(model, tuple) else model),
                "n": n,
                "english_win_rate": round(en / n, 4),
                "english_win_rate_ci_low": round(en_low, 4),
                "english_win_rate_ci_high": round(en_high, 4),
                "chinese_win_rate": round(zh / n, 4),
                "chinese_win_rate_ci_low": round(zh_low, 4),
                "chinese_win_rate_ci_high": round(zh_high, 4),
                "tie_rate": round(tie / n, 4),
                "tie_rate_ci_low": round(tie_low, 4),
                "tie_rate_ci_high": round(tie_high, 4),
                "english_share_among_non_ties": round(en / non_tie, 4) if non_tie else "",
            }
        )
    return out


def metric_quality_by_source(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (model, source, lang), group in sorted(group_by(rows, ("model", "source", "language")).items()):
        valid = [r for r in group if r.get("normalized_choice") in {"1", "2", "Tie"}]
        if not valid:
            continue
        half = sum(
            1.0 if r["normalized_choice"] == str(r["gold_label"]) else 0.5 if r["normalized_choice"] == "Tie" else 0.0
            for r in valid
        ) / len(valid)
        out.append(
            {
                "model": model,
                "source": source,
                "language": lang,
                "n": len(valid),
                "accuracy_tie_half": round(half, 4),
            }
        )
    return out


def metric_flip_by_source(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    quality = [r for r in rows if r["task_type"] == "quality" and r.get("normalized_choice") in {"1", "2", "Tie"}]
    keys = sorted({(r["model"], r["source"], r["language"]) for r in quality if r["language"] in {"zh", "cs"}})
    for model, source, target_lang in keys:
        pairs = []
        subset = [
            r
            for r in quality
            if r["model"] == model and r["source"] == source and r["language"] in {"en", target_lang}
        ]
        for (_item_id, _order), group in group_by(subset, ("item_id", "order")).items():
            by_lang = {r["language"]: r for r in group}
            if "en" in by_lang and target_lang in by_lang:
                pairs.append((by_lang["en"], by_lang[target_lang]))
        if not pairs:
            continue
        flips = sum(1 for en, other in pairs if en["normalized_choice"] != other["normalized_choice"])
        out.append(
            {
                "model": model,
                "source": source,
                "comparison": f"en_vs_{target_lang}",
                "n_pairs": len(pairs),
                "language_invariance_flip_rate": round(flips / len(pairs), 4),
            }
        )
    return out


def variant_high_risk_reasons(row: Dict[str, Any]) -> List[str]:
    """Conservative mechanical checks for transformations likely to need review."""
    reasons: List[str] = []
    number_re = re.compile(r"\d+(?:\.\d+)?")
    for lang in ("zh", "cs"):
        variant = row.get(lang, {})
        for field in ("input", "output_1", "output_2"):
            en_text = str(row.get("en", {}).get(field, ""))
            var_text = str(variant.get(field, ""))
            if not var_text.strip():
                reasons.append(f"{lang}:{field}:empty")
                continue
            en_numbers = number_re.findall(en_text)
            var_numbers = number_re.findall(var_text)
            length_ratio = len(var_text) / max(1, len(en_text))
            if len(en_numbers) >= 2 and len(var_numbers) == 0:
                reasons.append(f"{lang}:{field}:lost_all_numbers")
            elif abs(len(en_numbers) - len(var_numbers)) >= 3:
                reasons.append(f"{lang}:{field}:number_count_delta_{len(en_numbers)}_{len(var_numbers)}")
            if lang == "zh" and length_ratio < 0.10:
                reasons.append(f"{lang}:{field}:short_ratio_{length_ratio:.2f}")
            if lang == "cs" and length_ratio < 0.35:
                reasons.append(f"{lang}:{field}:short_ratio_{length_ratio:.2f}")
    return reasons


def metric_variant_audit(variant_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    unique_rows = list({row["id"]: row for row in variant_rows if row.get("status") == "ok"}.values())
    total = len(unique_rows)
    complete = sum(
        1
        for row in unique_rows
        if all(row.get(lang, {}).get(field) for lang in ("en", "zh", "cs") for field in ("input", "output_1", "output_2"))
    )
    manual = sum(1 for row in unique_rows if str(row.get("variant_model", "")).startswith("manual"))
    high_risk: Dict[str, List[str]] = {}
    for row in unique_rows:
        reasons = variant_high_risk_reasons(row)
        if reasons:
            high_risk[row["id"]] = reasons
    by_source = group_by(unique_rows, ("source",))
    rows = [
        {"check": "Unique transformed items", "count": total, "rate": 1.0 if total else float("nan")},
        {"check": "Complete EN/ZH/LS fields", "count": complete, "rate": complete / total if total else float("nan")},
        {"check": "Manual JSON repair", "count": manual, "rate": manual / total if total else float("nan")},
        {
            "check": "Mechanically flagged high-risk items",
            "count": len(high_risk),
            "rate": len(high_risk) / total if total else float("nan"),
        },
        {
            "check": "Items retained in sensitivity analysis",
            "count": total - len(high_risk),
            "rate": (total - len(high_risk)) / total if total else float("nan"),
        },
    ]
    for source, group in sorted(by_source.items()):
        source_name = source[0] if isinstance(source, tuple) else source
        flagged = sum(1 for row in group if row["id"] in high_risk)
        rows.append(
            {
                "check": f"High-risk items in {source_name}",
                "count": flagged,
                "rate": flagged / len(group) if group else float("nan"),
            }
        )
    return rows, sorted(high_risk)


def metric_filtered_invariance(rows: List[Dict[str, Any]], excluded_item_ids: List[str]) -> List[Dict[str, Any]]:
    excluded = set(excluded_item_ids)
    return metric_invariance([row for row in rows if row.get("item_id") not in excluded])


def metric_filtered_quality(rows: List[Dict[str, Any]], excluded_item_ids: List[str]) -> List[Dict[str, Any]]:
    excluded = set(excluded_item_ids)
    return metric_quality([row for row in rows if row.get("item_id") not in excluded])


def metric_paired_significance(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    quality = [r for r in rows if r["task_type"] == "quality" and r.get("normalized_choice") in {"1", "2", "Tie"}]
    for (model, target_lang), _ in sorted({(r["model"], r["language"]): True for r in quality if r["language"] in {"zh", "cs"}}.items()):
        pairs = []
        for (_item_id, _order), group in group_by(
            [r for r in quality if r["model"] == model and r["language"] in {"en", target_lang}],
            ("item_id", "order"),
        ).items():
            by_lang = {r["language"]: r for r in group}
            if "en" in by_lang and target_lang in by_lang:
                pairs.append((by_lang["en"], by_lang[target_lang]))
        if not pairs:
            continue
        en_only = 0
        target_only = 0
        both_correct = 0
        both_wrong = 0
        for en, other in pairs:
            en_correct = en["normalized_choice"] == str(en["gold_label"])
            other_correct = other["normalized_choice"] == str(other["gold_label"])
            if en_correct and other_correct:
                both_correct += 1
            elif en_correct and not other_correct:
                en_only += 1
            elif other_correct and not en_correct:
                target_only += 1
            else:
                both_wrong += 1
        discordant = en_only + target_only
        out.append(
            {
                "model": model,
                "model_short": short_model_name(model),
                "comparison": f"en_vs_{target_lang}",
                "n_pairs": len(pairs),
                "both_correct": both_correct,
                "english_only_correct": en_only,
                "target_only_correct": target_only,
                "both_wrong_or_tie": both_wrong,
                "mcnemar_exact_p": round(binomial_two_sided_p(min(en_only, target_only), discordant), 6)
                if discordant
                else 1.0,
            }
        )
    return out


def usage_report(rows: List[Dict[str, Any]], variant_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, float]] = {}

    def add(model: str, prompt: float, completion: float, calls: int) -> None:
        b = buckets.setdefault(model, {"calls": 0, "prompt_tokens": 0.0, "completion_tokens": 0.0})
        b["calls"] += calls
        b["prompt_tokens"] += prompt
        b["completion_tokens"] += completion

    for r in rows:
        if r.get("status") != "ok":
            continue
        usage = r.get("usage") or {}
        prompt = usage.get("prompt_tokens") or r.get("prompt_tokens_approx") or 0
        completion = usage.get("completion_tokens") or r.get("completion_tokens_approx") or 0
        add(r["model"], float(prompt), float(completion), 1)
    for r in variant_rows:
        if r.get("status") != "ok":
            continue
        usage = r.get("usage") or {}
        prompt = usage.get("prompt_tokens") or r.get("prompt_tokens_approx") or 0
        completion = usage.get("completion_tokens") or r.get("completion_tokens_approx") or 0
        add(f"{r.get('variant_model', DEFAULT_MODEL)} (variant-generation)", float(prompt), float(completion), 1)

    report = []
    for model, b in sorted(buckets.items()):
        base_model = model.replace(" (variant-generation)", "")
        pin, pout = PRICE_USD_PER_M.get(base_model, (None, None))
        cost = ""
        if pin is not None and pout is not None:
            cost = (b["prompt_tokens"] / 1_000_000) * pin + (b["completion_tokens"] / 1_000_000) * pout
            cost = round(cost, 4)
        report.append(
            {
                "model": model,
                "calls": int(b["calls"]),
                "prompt_tokens": int(b["prompt_tokens"]),
                "completion_tokens": int(b["completion_tokens"]),
                "estimated_cost_usd_reference_price": cost,
            }
        )
    return report


def make_figures(quality_rows: List[Dict[str, Any]], invariance_rows: List[Dict[str, Any]], tie_rows: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style(plt)

    if quality_rows:
        langs = ["en", "zh", "cs"]
        models = sorted({r["model"] for r in quality_rows})
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        width = GROUP_BAR_WIDTH
        x = list(range(len(models)))
        for i, lang in enumerate(langs):
            vals = []
            for model in models:
                row = next((r for r in quality_rows if r["model"] == model and r["language"] == lang), None)
                vals.append(100 * float(row["accuracy_tie_half"]) if row else 0)
            ax.bar(
                [v + (i - 1) * width for v in x],
                vals,
                width=width,
                label=display_lang(lang),
                color=LANG_COLORS[lang],
                **BAR_EDGE,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([short_model_name(m) for m in models], rotation=18, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Tie-half accuracy (%)")
        ax.set_title("LLMBar judge accuracy by language condition")
        ax.legend(frameon=False)
        style_bar_axis(ax)
        fig.tight_layout()
        fig.savefig(FIGURES / "accuracy_by_language.png", dpi=240)
        fig.savefig(FIGURES / "accuracy_by_language.pdf")
        plt.close(fig)

    if invariance_rows:
        labels = [f"{r.get('model_short', short_model_name(r['model']))}\n{display_comparison(r['comparison'])}" for r in invariance_rows]
        vals = [100 * float(r["language_invariance_flip_rate"]) for r in invariance_rows]
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ax.bar(labels, vals, width=0.58, color="#4C78A8", **BAR_EDGE)
        ax.set_ylim(0, max(10, min(100, max(vals) + 10)))
        ax.set_ylabel("Judgment-flip rate vs. EN (%)")
        ax.set_title("Judgment changes after language transformation")
        ax.tick_params(axis="x", labelrotation=20)
        style_bar_axis(ax)
        fig.tight_layout()
        fig.savefig(FIGURES / "language_invariance_flip_rate.png", dpi=240)
        fig.savefig(FIGURES / "language_invariance_flip_rate.pdf")
        plt.close(fig)

    if tie_rows:
        models = [r.get("model_short", short_model_name(r["model"])) for r in tie_rows]
        en = [100 * float(r["english_win_rate"]) for r in tie_rows]
        zh = [100 * float(r["chinese_win_rate"]) for r in tie_rows]
        tie = [100 * float(r["tie_rate"]) for r in tie_rows]
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        x = list(range(len(models)))
        ax.bar(x, en, width=0.58, label="EN wins", color=LANG_COLORS["en"], **BAR_EDGE)
        ax.bar(x, zh, width=0.58, bottom=en, label="ZH wins", color=LANG_COLORS["zh"], **BAR_EDGE)
        ax.bar(x, tie, width=0.58, bottom=[a + b for a, b in zip(en, zh)], label="Tie", color="#BDBDBD", **BAR_EDGE)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Tie-probe outcomes (%)")
        ax.set_title("Translation-equivalent tie-probe outcomes")
        ax.legend(frameon=False, ncols=3, fontsize=8)
        style_bar_axis(ax)
        fig.tight_layout()
        fig.savefig(FIGURES / "tie_probe_language_preference.png", dpi=240)
        fig.savefig(FIGURES / "tie_probe_language_preference.pdf")
        plt.close(fig)


def make_main_figure(
    quality_rows: List[Dict[str, Any]],
    invariance_rows: List[Dict[str, Any]],
    tie_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    apply_plot_style(plt)

    def box(ax: Any, xy: Tuple[float, float], w: float, h: float, title: str, body: str, fc: str) -> None:
        patch = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(xy[0] + w / 2, xy[1] + h - 0.055, title, ha="center", va="top", fontsize=10.2, fontweight="bold")
        ax.text(xy[0] + w / 2, xy[1] + h * 0.39, body, ha="center", va="center", fontsize=8.2, linespacing=1.13)

    def arrow(ax: Any, x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#444444",
            )
        )

    en_acc = mean([float(r["accuracy_tie_half"]) for r in quality_rows if r["language"] == "en"])
    zh_acc = mean([float(r["accuracy_tie_half"]) for r in quality_rows if r["language"] == "zh"])
    cs_acc = mean([float(r["accuracy_tie_half"]) for r in quality_rows if r["language"] == "cs"])
    flip_vals = [float(r["language_invariance_flip_rate"]) for r in invariance_rows]
    tie_vals = [float(r["tie_rate"]) for r in tie_rows]
    high_risk = next((r for r in audit_rows if r["check"] == "Mechanically flagged high-risk items"), None)
    retained = next((r for r in audit_rows if r["check"] == "Items retained in sensitivity analysis"), None)

    fig, ax = plt.subplots(figsize=(7.5, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.02, 0.56), 0.20, 0.31, "LLMBar data", "419 pairwise items\nNatural + 4 adversarial\nobjective gold labels", "#EEF3FA")
    box(ax, (0.29, 0.56), 0.20, 0.31, "Language variants", "same item + label\nEN original\nZH translation\nLS mixed-language", "#F7F0E8")
    box(ax, (0.56, 0.56), 0.20, 0.31, "Judge metrics", "4 API judges\naccuracy + flip rates\norder + tie checks", "#EFF6EF")
    box(
        ax,
        (0.78, 0.56),
        0.20,
        0.31,
        "Diagnostics",
        f"EN acc. {100*en_acc:.1f}%\nZH/LS acc. {100*zh_acc:.1f}/{100*cs_acc:.1f}%\nflip {100*min(flip_vals):.1f}-{100*max(flip_vals):.1f}%",
        "#F2EFF8",
    )

    arrow(ax, 0.22, 0.72, 0.29, 0.72)
    arrow(ax, 0.49, 0.72, 0.56, 0.72)
    arrow(ax, 0.76, 0.72, 0.78, 0.72)

    box(
        ax,
        (0.12, 0.10),
        0.24,
        0.27,
        "Transformation audit",
        f"complete fields: 419/419\nflagged high-risk: {high_risk['count'] if high_risk else 0}\nretained: {retained['count'] if retained else 0}",
        "#FAF5E8",
    )
    box(
        ax,
        (0.39, 0.10),
        0.24,
        0.27,
        "Preference stability",
        "same item/order\nEN vs. ZH/LS\npaired CIs + tests",
        "#EAF5F7",
    )
    box(
        ax,
        (0.66, 0.10),
        0.24,
        0.27,
        "Language preference",
        f"mostly Tie\nmean tie rate {100*mean(tie_vals):.1f}%\nnon-ties do not favor EN",
        "#F6ECF1",
    )
    arrow(ax, 0.66, 0.56, 0.51, 0.37)
    arrow(ax, 0.88, 0.56, 0.78, 0.37)
    arrow(ax, 0.32, 0.56, 0.25, 0.37)

    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES / "main_figure.png", dpi=260)
    fig.savefig(FIGURES / "main_figure.pdf")
    plt.close(fig)


def make_extra_figures(
    quality_rows: List[Dict[str, Any]],
    invariance_rows: List[Dict[str, Any]],
    position_rows: List[Dict[str, Any]],
    source_quality_rows: List[Dict[str, Any]],
    source_flip_rows: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style(plt)

    models = sorted({r["model"] for r in quality_rows})
    short = {
        "gpt-4.1-mini": "GPT-4.1 Mini",
        "claude-haiku-4-5-20251001": "Claude Haiku",
        "gemini-2.5-flash": "Gemini Flash",
        "deepseek-v4-flash": "DeepSeek Flash",
    }

    if quality_rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        x = list(range(len(models)))
        width = GROUP_BAR_WIDTH
        zhd, csd = [], []
        for model in models:
            by_lang = {r["language"]: float(r["accuracy_tie_half"]) for r in quality_rows if r["model"] == model}
            zhd.append(100 * (by_lang.get("en", 0) - by_lang.get("zh", 0)))
            csd.append(100 * (by_lang.get("en", 0) - by_lang.get("cs", 0)))
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar([i - width / 2 for i in x], zhd, width=width, label="EN - ZH", color=LANG_COLORS["zh"], **BAR_EDGE)
        ax.bar([i + width / 2 for i in x], csd, width=width, label="EN - LS", color=LANG_COLORS["cs"], **BAR_EDGE)
        ax.set_xticks(x)
        ax.set_xticklabels([short.get(m, m) for m in models], rotation=18, ha="right")
        ax.set_ylabel("Tie-half accuracy drop (points)")
        ax.set_title("Accuracy loss relative to EN")
        ax.legend(frameon=False)
        style_bar_axis(ax)
        fig.tight_layout()
        fig.savefig(FIGURES / "accuracy_drop_relative_to_english.png", dpi=240)
        plt.close(fig)

    if position_rows:
        langs = ["en", "zh", "cs"]
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        x = list(range(len(models)))
        width = GROUP_BAR_WIDTH
        for idx, lang in enumerate(langs):
            vals = []
            for model in models:
                row = next((r for r in position_rows if r["model"] == model and r["language"] == lang), None)
                vals.append(100 * float(row["position_inconsistency_rate"]) if row else 0)
            ax.bar(
                [v + (idx - 1) * width for v in x],
                vals,
                width=width,
                color=LANG_COLORS[lang],
                label=display_lang(lang),
                **BAR_EDGE,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([short.get(m, m) for m in models], rotation=18, ha="right")
        ax.set_ylabel("Position inconsistency (%)")
        ax.set_title("Answer-order sensitivity by language")
        ax.legend(frameon=False, ncols=3)
        style_bar_axis(ax)
        fig.tight_layout()
        fig.savefig(FIGURES / "position_inconsistency_by_language.png", dpi=240)
        plt.close(fig)

    if source_quality_rows:
        sources = sorted({r["source"] for r in source_quality_rows})
        langs = ["en", "zh", "cs"]
        matrix = []
        for source in sources:
            row_vals = []
            for lang in langs:
                vals = [float(r["accuracy_tie_half"]) for r in source_quality_rows if r["source"] == source and r["language"] == lang]
                row_vals.append(100 * mean(vals))
            matrix.append(row_vals)
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=55, vmax=95)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([display_lang(l) for l in langs])
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("Mean tie-half accuracy by LLMBar subset")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy (%)")
        fig.tight_layout()
        fig.savefig(FIGURES / "source_language_accuracy_heatmap.png", dpi=240)
        plt.close(fig)

    if source_flip_rows:
        sources = sorted({r["source"] for r in source_flip_rows})
        matrix = []
        for source in sources:
            row_vals = []
            for model in models:
                vals = [float(r["language_invariance_flip_rate"]) for r in source_flip_rows if r["source"] == source and r["model"] == model]
                row_vals.append(100 * mean(vals))
            matrix.append(row_vals)
        fig, ax = plt.subplots(figsize=(7.4, 3.7))
        im = ax.imshow(matrix, cmap="OrRd", vmin=0, vmax=max(25, max(max(r) for r in matrix)))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([short.get(m, m) for m in models], rotation=18, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("Mean judgment-flip rate by LLMBar subset")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Judgment-flip rate (%)")
        fig.tight_layout()
        fig.savefig(FIGURES / "source_flip_heatmap.png", dpi=240)
        plt.close(fig)


def latex_escape(s: Any) -> str:
    text = str(s)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def write_latex_tables(
    quality_rows: List[Dict[str, Any]],
    invariance_rows: List[Dict[str, Any]],
    tie_rows: List[Dict[str, Any]],
    usage_rows: List[Dict[str, Any]],
    position_rows: List[Dict[str, Any]],
    source_quality_rows: List[Dict[str, Any]],
    source_flip_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    filtered_invariance_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"Model & \shortstack{Language\\condition} & N & \shortstack{Strict\\acc. (\%)} & \shortstack{Tie-half\\acc. (\%)} & \shortstack{Tie-half\\95\% CI} & \shortstack{Position-A\\rate (\%)} & \shortstack{Tie\\rate (\%)} \\"
    )
    lines.append(r"\hline")
    for r in quality_rows:
        lines.append(
            f"{latex_escape(r.get('model_short', short_model_name(r['model'])))} & {latex_escape(display_lang(r['language']))} & {r['n']} & "
            f"{fmt_pct(r['accuracy_strict'])} & {fmt_pct(r['accuracy_tie_half'])} & "
            f"{latex_escape(fmt_ci(r['accuracy_tie_half_ci_low'], r['accuracy_tie_half_ci_high']))} & "
            f"{fmt_pct(r['position_A_rate'])} & {fmt_pct(r['tie_rate'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Quality-judgment accuracy by language condition on LLMBar. Strict accuracy counts ties as incorrect; tie-half gives ties half credit. Position-A is choosing the displayed first answer before order normalization; all rates are percentages.}"
    )
    lines.append(r"\label{tab:quality}")
    lines.append(r"\end{table*}")

    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"Model & \shortstack{Language\\pair} & N & \shortstack{Judgment\\flip (\%)} & \shortstack{Judgment flip\\95\% CI} & \shortstack{Correctness\\flip (\%)} & \shortstack{Correctness flip\\95\% CI} \\"
    )
    lines.append(r"\hline")
    for r in invariance_rows:
        lines.append(
            f"{latex_escape(r.get('model_short', short_model_name(r['model'])))} & {latex_escape(display_comparison(r['comparison']))} & {r['n_pairs']} & "
            f"{fmt_pct(r['language_invariance_flip_rate'])} & "
            f"{latex_escape(fmt_ci(r['language_invariance_flip_rate_ci_low'], r['language_invariance_flip_rate_ci_high']))} & "
            f"{fmt_pct(r['gold_correctness_flip_rate'])} & "
            f"{latex_escape(fmt_ci(r['gold_correctness_flip_rate_ci_low'], r['gold_correctness_flip_rate_ci_high']))} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Language-invariance flip rates relative to English. Judgment flip counts any changed normalized preference; correctness flip counts a change between strict gold-correct and not strict gold-correct. All rates and intervals are percentages.}")
    lines.append(r"\label{tab:invariance}")
    lines.append(r"\end{table*}")

    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Model & N & \shortstack{EN wins\\(\%)} & \shortstack{ZH wins\\(\%)} & \shortstack{Tie\\(\%)} & \shortstack{EN share\\non-tie (\%)} \\")
    lines.append(r"\hline")
    for r in tie_rows:
        lines.append(
            f"{latex_escape(r.get('model_short', short_model_name(r['model'])))} & {r['n']} & {fmt_pct(r['english_win_rate'])} & "
            f"{fmt_pct(r['chinese_win_rate'])} & {fmt_pct(r['tie_rate'])} & "
            f"{fmt_pct(r['english_share_among_non_ties']) if r['english_share_among_non_ties'] != '' else '--'} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Language preference in translation-equivalent tie probes. EN/ZH win rates indicate which language version is preferred; EN share is computed over non-tie decisions. All rates are percentages.}")
    lines.append(r"\label{tab:tieprobe}")
    lines.append(r"\end{table}")

    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\hline")
    lines.append(r"Model & Calls & \shortstack{Input\\tokens} & \shortstack{Output\\tokens} \\")
    lines.append(r"\hline")
    for r in usage_rows:
        if int(r["prompt_tokens"]) == 0 and int(r["completion_tokens"]) == 0:
            continue
        lines.append(
            f"{latex_escape(short_model_name(r['model']))} & {r['calls']} & {int(r['prompt_tokens']):,} & {int(r['completion_tokens']):,} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Observed or approximated API usage. Costs depend on the endpoint's billing plan.}")
    lines.append(r"\label{tab:usage}")
    lines.append(r"\end{table}")

    (PAPER / "tables.tex").write_text("\n".join(lines), encoding="utf-8")

    extended: List[str] = []
    sources = sorted({r["source"] for r in source_quality_rows})
    extended.append(r"\begin{table*}[t]")
    extended.append(r"\centering")
    extended.append(r"\small")
    extended.append(r"\setlength{\tabcolsep}{6pt}")
    extended.append(r"\renewcommand{\arraystretch}{1.08}")
    extended.append(r"\begin{tabular}{lrrrrr}")
    extended.append(r"\hline")
    extended.append(
        r"Subset & \shortstack{EN\\acc. (\%)} & \shortstack{ZH\\acc. (\%)} & \shortstack{LS\\acc. (\%)} & \shortstack{EN--ZH\\judgment flip (\%)} & \shortstack{EN--LS\\judgment flip (\%)} \\"
    )
    extended.append(r"\hline")
    for source in sources:
        acc = {}
        for lang in ("en", "zh", "cs"):
            vals = [float(r["accuracy_tie_half"]) for r in source_quality_rows if r["source"] == source and r["language"] == lang]
            acc[lang] = mean(vals)
        flip = {}
        for comp in ("en_vs_zh", "en_vs_cs"):
            vals = [float(r["language_invariance_flip_rate"]) for r in source_flip_rows if r["source"] == source and r["comparison"] == comp]
            flip[comp] = mean(vals)
        extended.append(
            f"{latex_escape(source)} & {fmt_pct(acc['en'])} & {fmt_pct(acc['zh'])} & {fmt_pct(acc['cs'])} & "
            f"{fmt_pct(flip['en_vs_zh'])} & {fmt_pct(flip['en_vs_cs'])} \\\\"
        )
    extended.append(r"\hline")
    extended.append(r"\end{tabular}")
    extended.append(r"\caption{Subset-level averages across the four judges. Accuracy uses tie-half scoring; judgment flip compares transformed language conditions with English. All rates are percentages.}")
    extended.append(r"\label{tab:source-aggregate}")
    extended.append(r"\end{table*}")

    extended.append("")
    extended.append(r"\begin{table}[t]")
    extended.append(r"\centering")
    extended.append(r"\small")
    extended.append(r"\setlength{\tabcolsep}{4pt}")
    extended.append(r"\renewcommand{\arraystretch}{1.08}")
    extended.append(r"\begin{tabular}{lrrr}")
    extended.append(r"\hline")
    extended.append(r"Model & \shortstack{EN\\(\%)} & \shortstack{ZH\\(\%)} & \shortstack{LS\\(\%)} \\")
    extended.append(r"\hline")
    models = sorted({r["model"] for r in position_rows})
    for model in models:
        vals = {}
        for lang in ("en", "zh", "cs"):
            row = next((r for r in position_rows if r["model"] == model and r["language"] == lang), None)
            vals[lang] = float(row["position_inconsistency_rate"]) if row else float("nan")
        extended.append(f"{latex_escape(short_model_name(model))} & {fmt_pct(vals['en'])} & {fmt_pct(vals['zh'])} & {fmt_pct(vals['cs'])} \\\\")
    extended.append(r"\hline")
    extended.append(
        f"Mean & {fmt_pct(mean([float(r['position_inconsistency_rate']) for r in position_rows if r['language']=='en']))} & "
        f"{fmt_pct(mean([float(r['position_inconsistency_rate']) for r in position_rows if r['language']=='zh']))} & "
        f"{fmt_pct(mean([float(r['position_inconsistency_rate']) for r in position_rows if r['language']=='cs']))} \\\\"
    )
    extended.append(r"\hline")
    extended.append(r"\end{tabular}")
    extended.append(r"\caption{Position inconsistency by model and language condition. A case is inconsistent when the normalized winner changes after swapping answer order; all rates are percentages.}")
    extended.append(r"\label{tab:position-extra}")
    extended.append(r"\end{table}")

    extended.append("")
    extended.append(r"\begin{table}[t]")
    extended.append(r"\centering")
    extended.append(r"\small")
    extended.append(r"\setlength{\tabcolsep}{4pt}")
    extended.append(r"\renewcommand{\arraystretch}{1.08}")
    extended.append(r"\begin{tabular}{p{0.68\linewidth}rr}")
    extended.append(r"\hline")
    extended.append(r"Audit check & Count & \shortstack{Rate\\(\%)} \\")
    extended.append(r"\hline")
    for r in audit_rows[:5]:
        extended.append(f"{latex_escape(r['check'])} & {r['count']} & {fmt_pct(r['rate'])} \\\\")
    extended.append(r"\hline")
    extended.append(r"\end{tabular}")
    extended.append(r"\caption{Automatic transformation audit. High-risk flags are conservative mechanical warnings based on empty fields, severe length shrinkage, or large numeric-token count changes; they are not treated as semantic labels.}")
    extended.append(r"\label{tab:variant-audit}")
    extended.append(r"\end{table}")

    extended.append("")
    extended.append(r"\begin{table*}[t]")
    extended.append(r"\centering")
    extended.append(r"\small")
    extended.append(r"\setlength{\tabcolsep}{6pt}")
    extended.append(r"\renewcommand{\arraystretch}{1.08}")
    extended.append(r"\begin{tabular}{llrrrr}")
    extended.append(r"\hline")
    extended.append(r"Model & \shortstack{Language\\pair} & N & \shortstack{Full judgment\\flip (\%)} & \shortstack{Filtered judgment\\flip (\%)} & \shortstack{Correctness\\test p} \\")
    extended.append(r"\hline")
    for full in invariance_rows:
        clean = next((r for r in filtered_invariance_rows if r["model"] == full["model"] and r["comparison"] == full["comparison"]), None)
        paired = next((r for r in paired_rows if r["model"] == full["model"] and r["comparison"] == full["comparison"]), None)
        pval = paired["mcnemar_exact_p"] if paired else ""
        ptext = f"{pval:.3g}" if isinstance(pval, float) else str(pval)
        extended.append(
            f"{latex_escape(short_model_name(full['model']))} & {latex_escape(display_comparison(full['comparison']))} & "
            f"{full['n_pairs']} & {fmt_pct(full['language_invariance_flip_rate'])} & "
            f"{fmt_pct(clean['language_invariance_flip_rate']) if clean else '--'} & {ptext} \\\\"
        )
    extended.append(r"\hline")
    extended.append(r"\end{tabular}")
    extended.append(r"\caption{Sensitivity and paired-significance checks. Filtered judgment flip excludes mechanically flagged high-risk transformation items. The p-value is an exact two-sided McNemar/binomial test on strict gold-correctness changes between English and the target language.}")
    extended.append(r"\label{tab:sensitivity-significance}")
    extended.append(r"\end{table*}")

    (PAPER / "extended_tables.tex").write_text("\n".join(extended), encoding="utf-8")


def cmd_analyze(args: argparse.Namespace) -> None:
    ensure_dirs()
    success_rows = [r for r in iter_jsonl(RESULTS / "judgments.jsonl") if r.get("status") == "ok"]
    by_request: Dict[str, Dict[str, Any]] = {}
    for row in success_rows:
        by_request[row["request_key"]] = row
    rows = list(by_request.values())

    variant_success = [r for r in iter_jsonl(PROCESSED / "variants.jsonl") if r.get("status") == "ok"]
    by_variant: Dict[str, Dict[str, Any]] = {}
    for row in variant_success:
        by_variant[row["id"]] = row
    variant_rows = list(by_variant.values())
    if not rows:
        raise RuntimeError("No successful judgments found.")
    for r in rows:
        if r["task_type"] == "quality":
            r["normalized_choice"] = normalized_choice_for_quality(r)
        elif r["task_type"] == "tie_probe":
            r["language_choice"] = tie_probe_language_choice(r)

    q_rows = metric_quality([r for r in rows if r["task_type"] == "quality"])
    inv_rows = metric_invariance(rows)
    pos_rows = metric_position(rows)
    tie_rows = metric_tie_probe(rows)
    source_quality_rows = metric_quality_by_source([r for r in rows if r["task_type"] == "quality"])
    source_flip_rows = metric_flip_by_source(rows)
    audit_rows, high_risk_item_ids = metric_variant_audit(variant_rows)
    filtered_quality_rows = metric_filtered_quality([r for r in rows if r["task_type"] == "quality"], high_risk_item_ids)
    filtered_inv_rows = metric_filtered_invariance(rows, high_risk_item_ids)
    paired_rows = metric_paired_significance(rows)
    usage_rows = usage_report(rows, variant_rows)

    write_csv(RESULTS / "quality_metrics.csv", q_rows)
    write_csv(RESULTS / "language_invariance_metrics.csv", inv_rows)
    write_csv(RESULTS / "position_metrics.csv", pos_rows)
    write_csv(RESULTS / "tie_probe_metrics.csv", tie_rows)
    write_csv(RESULTS / "source_quality_metrics.csv", source_quality_rows)
    write_csv(RESULTS / "source_flip_metrics.csv", source_flip_rows)
    write_csv(RESULTS / "variant_audit_metrics.csv", audit_rows)
    write_csv(RESULTS / "filtered_quality_metrics.csv", filtered_quality_rows)
    write_csv(RESULTS / "filtered_language_invariance_metrics.csv", filtered_inv_rows)
    write_csv(RESULTS / "paired_significance_metrics.csv", paired_rows)
    write_csv(RESULTS / "usage_report.csv", usage_rows)
    write_json(
        RESULTS / "summary.json",
        {
            "quality": q_rows,
            "language_invariance": inv_rows,
            "position": pos_rows,
            "tie_probe": tie_rows,
            "source_quality": source_quality_rows,
            "source_flip": source_flip_rows,
            "variant_audit": audit_rows,
            "high_risk_item_ids": high_risk_item_ids,
            "filtered_quality": filtered_quality_rows,
            "filtered_language_invariance": filtered_inv_rows,
            "paired_significance": paired_rows,
            "usage": usage_rows,
        },
    )
    make_main_figure(q_rows, inv_rows, tie_rows, audit_rows)
    make_figures(q_rows, inv_rows, tie_rows)
    make_extra_figures(q_rows, inv_rows, pos_rows, source_quality_rows, source_flip_rows)
    write_latex_tables(
        q_rows,
        inv_rows,
        tie_rows,
        usage_rows,
        pos_rows,
        source_quality_rows,
        source_flip_rows,
        audit_rows,
        filtered_inv_rows,
        paired_rows,
    )
    print(f"Wrote metrics to {RESULTS}")
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote LaTeX tables to {PAPER / 'tables.tex'} and {PAPER / 'extended_tables.tex'}")
    print(f"Analyzed {len(rows)} unique successful judgments and {len(variant_rows)} unique variants.")


def cmd_budget(args: argparse.Namespace) -> None:
    ensure_dirs()
    n_items = args.items
    calls_per_item = 8
    judge_calls = n_items * calls_per_item
    prompt_tokens = judge_calls * args.prompt_tokens
    completion_tokens = judge_calls * args.completion_tokens
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    total = 0.0
    for model in models:
        pin, pout = PRICE_USD_PER_M.get(model, (None, None))
        cost = ""
        if pin is not None and pout is not None:
            cost_val = prompt_tokens / 1_000_000 * pin + completion_tokens / 1_000_000 * pout
            cost = round(cost_val, 4)
            total += cost_val
        rows.append(
            {
                "model": model,
                "items": n_items,
                "calls": judge_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": cost,
            }
        )
    write_csv(RESULTS / "budget_estimate.csv", rows)
    print(json.dumps({"rows": rows, "known_price_total_usd": round(total, 4)}, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Judge-LS experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Sample LLMBar items.")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_prepare)

    def add_api_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--api-base", default=DEFAULT_API_BASE)
        p.add_argument("--timeout", type=int, default=120)
        p.add_argument("--retries", type=int, default=3)
        p.add_argument("--sleep", type=float, default=0.2)
        p.add_argument("--max-tokens", type=int, default=900)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("variants", help="Generate zh and language-switched variants.")
    add_api_args(p)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_variants)

    p = sub.add_parser("judge", help="Run LLM judge calls.")
    add_api_args(p)
    p.add_argument("--models", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tasks", type=int, default=0)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("analyze", help="Summarize results and make figures.")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("budget", help="Estimate token usage and rough bills.")
    p.add_argument("--items", type=int, default=200)
    p.add_argument("--models", default="gpt-5.4-mini,claude-haiku-4-5,gemini-2.5-flash,deepseek-v4-flash")
    p.add_argument("--prompt-tokens", type=int, default=1100)
    p.add_argument("--completion-tokens", type=int, default=80)
    p.set_defaults(func=cmd_budget)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    ensure_dirs()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
