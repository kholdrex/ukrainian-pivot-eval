"""Multiple-choice items: parallel Global-MMLU questions (en/uk) and native Ukrainian ZNO history questions."""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

HF = "https://huggingface.co/datasets"
GMMLU_REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
ZNO_REVISION = "473616f29e9f6504593235deb7aa27e85b400c58"
GMMLU_FILES = {lang: f"{lang}/test-00000-of-00001.parquet" for lang in ("en", "uk")}
ZNO_FILES = ("train.jsonl", "test.jsonl")
LETTERS = "ABCD"
EMPTY_TABLE_RE = re.compile(r"(\|[\s|-]*\|\s*)+$")


@dataclass(frozen=True)
class Item:
    id: str
    source: str
    category: str
    cultural: str
    question_uk: str
    options_uk: tuple[str, ...]
    answer: int
    question_en: str | None = None
    options_en: tuple[str, ...] | None = None


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def download(root: Path) -> None:
    for lang, name in GMMLU_FILES.items():
        fetch(f"{HF}/CohereLabs/Global-MMLU/resolve/{GMMLU_REVISION}/{name}", root / "gmmlu" / f"{lang}.parquet")
    for name in ZNO_FILES:
        fetch(f"{HF}/osyvokon/zno/resolve/{ZNO_REVISION}/{name}", root / "zno" / name)


def gmmlu(root: Path) -> list[Item]:
    en = pd.read_parquet(root / "gmmlu" / "en.parquet").set_index("sample_id")
    uk = pd.read_parquet(root / "gmmlu" / "uk.parquet").set_index("sample_id")
    columns = ["option_a", "option_b", "option_c", "option_d"]
    items = []
    for sid, u in uk.iterrows():
        e = en.loc[sid]
        if u["answer"] != e["answer"]:
            raise ValueError(f"answer mismatch for {sid}")
        items.append(
            Item(
                id=f"gmmlu/{sid}",
                source="gmmlu",
                category=u["subject_category"],
                cultural=u["cultural_sensitivity_label"],
                question_uk=u["question"],
                options_uk=tuple(str(u[c]) for c in columns),
                answer=LETTERS.index(u["answer"]),
                question_en=e["question"],
                options_en=tuple(str(e[c]) for c in columns),
            )
        )
    return items


def zno(root: Path) -> list[Item]:
    items = []
    for name in ZNO_FILES:
        with (root / "zno" / name).open() as f:
            rows = [json.loads(line) for line in f]
        for k, r in enumerate(rows):
            if r["subject"] != "history-of-ukraine":
                continue
            markers = [a["marker"] for a in r["answers"]]
            items.append(
                Item(
                    id=f"zno/{name.split('.')[0]}/{k}",
                    source="zno",
                    category="History of Ukraine",
                    cultural="native",
                    question_uk=r["question"],
                    options_uk=tuple(a["text"] for a in r["answers"]),
                    answer=markers.index(r["correct_answers"][0]),
                )
            )
    return items


def lost_image(item: Item) -> bool:
    """The question ends with an empty table, the remains of a picture that the dataset could not convert."""
    return EMPTY_TABLE_RE.search(item.question_uk.strip()) is not None and "| --- |" in item.question_uk


def stratified(items: list[Item], per_category: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    by_category: dict[str, list[Item]] = {}
    for it in items:
        by_category.setdefault(it.category, []).append(it)
    chosen = [it for c in sorted(by_category) for it in rng.sample(by_category[c], per_category)]
    return sorted(chosen, key=lambda it: it.id)
