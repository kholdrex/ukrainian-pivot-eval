"""Translation and answering calls for every model, condition and item; stored as JSONL and resumable."""

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

from . import prompts
from .client import OpenRouter
from .data import Item

MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-12b-it",
    "mistralai/mistral-small-3.2-24b-instruct",
]
EXTERNAL_TRANSLATOR = "openai/gpt-4o-mini"
PROVIDERS = {
    "meta-llama/llama-3.2-3b-instruct": "parasail",
    "qwen/qwen-2.5-7b-instruct": "phala",
    "meta-llama/llama-3.1-8b-instruct": "deepinfra",
    "google/gemma-3-12b-it": "deepinfra",
    "mistralai/mistral-small-3.2-24b-instruct": "deepinfra",
    EXTERNAL_TRANSLATOR: "openai",
}
CONDITIONS = ("uk", "en", "self", "ext")
ANSWER_TOKENS = 16
TRANSLATION_TOKENS = 1024
WORKERS = 32


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


class Job(NamedTuple):
    kind: str
    model: str
    condition: str
    item: Item

    def key(self) -> tuple:
        return self.kind, self.model, self.condition, self.item.id


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows = {key(row): row for row in load_rows(path)}

    def add(self, row: dict) -> None:
        with self.lock:
            self.rows[key(row)] = row
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row: dict) -> tuple:
    return row["kind"], row["model"], row["condition"], row["id"]


def conditions(item: Item) -> list[str]:
    return [c for c in CONDITIONS if c != "en" or item.question_en is not None]


def translation_jobs(models: list[str], items: list[Item]) -> list[Job]:
    return [Job("translate", t, "uk", it) for t in [*models, EXTERNAL_TRANSLATOR] for it in items]


def source_text(job: Job) -> str:
    return prompts.block(job.item.question_uk, job.item.options_uk)


def answer_text(store: Store, job: Job) -> str:
    item = job.item
    if job.condition == "uk":
        return prompts.block(item.question_uk, item.options_uk)
    if job.condition == "en":
        return prompts.block(item.question_en, item.options_en)
    translator = job.model if job.condition == "self" else EXTERNAL_TRANSLATOR
    return store.rows[("translate", translator, "uk", item.id)]["text"]


def execute(client: OpenRouter, store: Store, job: Job, text: str) -> None:
    translate = job.kind == "translate"
    messages = prompts.translate_messages(text) if translate else prompts.answer_messages(text)
    limit = TRANSLATION_TOKENS if translate else ANSWER_TOKENS
    try:
        reply = client.chat(job.model, messages, limit, PROVIDERS[job.model])
    except RuntimeError:
        return
    store.add({"kind": job.kind, "model": job.model, "condition": job.condition, "id": job.item.id, **asdict(reply)})


def missing(jobs: list[Job], store: Store) -> list[Job]:
    return [j for j in jobs if j.key() not in store.rows]


def run_all(jobs: list[Job], client: OpenRouter, store: Store, text: Callable[[Job], str]) -> None:
    """Run the jobs; calls that still fail after retries are repeated in later passes while any pass makes progress."""
    todo = missing(jobs, store)
    while todo:
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(lambda j: execute(client, store, j, text(j)), todo))
        left = missing(jobs, store)
        if len(left) == len(todo):
            raise RuntimeError(f"{len(left)} calls failed repeatedly")
        todo = left


def run(models: list[str], items: list[Item], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    client, store = OpenRouter(), Store(out)
    run_all(translation_jobs(models, items), client, store, source_text)
    jobs = [Job("answer", m, c, it) for m in models for it in items for c in conditions(it)]
    run_all(jobs, client, store, lambda j: answer_text(store, j))
