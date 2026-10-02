"""Accuracy by condition, paired contrasts, transitions, costs and figures from the stored calls."""

import json
from math import comb
from pathlib import Path

import numpy as np

from . import prompts
from .data import Item, lost_image
from .experiment import EXTERNAL_TRANSLATOR, MODELS, load_rows

SEED = 2026
RESAMPLES = 2000
LABELS = {
    "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B",
    "qwen/qwen-2.5-7b-instruct": "Qwen 2.5 7B",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "google/gemma-3-12b-it": "Gemma 3 12B",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 24B",
}
SOURCES = {"gmmlu": "Global-MMLU", "zno": "ZNO history"}
CONTRASTS = (("self", "uk"), ("ext", "uk"), ("en", "uk"), ("ext", "self"))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def interval(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def outcomes(rows: list[dict], items: dict[str, Item]) -> dict:
    """correct[(model, condition)][item id] and invalid replies."""
    correct, invalid = {}, {}
    for r in rows:
        if r["kind"] != "answer" or r["id"] not in items:
            continue
        choice = prompts.parse_answer(r["text"])
        correct.setdefault((r["model"], r["condition"]), {})[r["id"]] = choice == items[r["id"]].answer
        invalid.setdefault((r["model"], r["condition"]), {})[r["id"]] = choice is None
    return {"correct": correct, "invalid": invalid}


def tokens(rows: list[dict], items: dict[str, Item]) -> dict:
    """Mean prompt + completion tokens per question for answering and for translating."""
    acc = {}
    for r in rows:
        if r["id"] not in items:
            continue
        key = (r["kind"], r["model"], r["condition"], items[r["id"]].source)
        acc.setdefault(key, []).append(r["prompt_tokens"] + r["completion_tokens"])
    return {"|".join(k): float(np.mean(v)) for k, v in acc.items()}


def first_occurrences(items: dict[str, Item], source: str) -> list[str]:
    """Item ids of a source with repeated questions (same question and options) kept once, in dataset order."""
    seen, out = set(), []
    for i, it in items.items():
        key = (it.question_uk.strip(), it.options_uk)
        if it.source == source and key not in seen:
            seen.add(key)
            out.append(i)
    return out


def analyse_source(source: str, models: list[str], data: dict, items: dict[str, Item],
                   ids: list[str] | None = None) -> tuple[dict, dict]:
    ids = sorted(ids if ids is not None else (i for i, it in items.items() if it.source == source))
    conditions = ["uk", "en", "self", "ext"] if source == "gmmlu" else ["uk", "self", "ext"]
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(ids), (RESAMPLES, len(ids)))
    out, pvalues = {}, {}
    for m in models:
        table = {c: np.array([data["correct"][(m, c)][i] for i in ids], dtype=float) for c in conditions}
        res = {"n": len(ids), "accuracy": {}, "invalid": {}, "contrast": {}}
        for c in conditions:
            res["accuracy"][c] = {"value": float(table[c].mean()), "ci": interval(table[c][boots].mean(axis=1))}
            res["invalid"][c] = float(np.mean([data["invalid"][(m, c)][i] for i in ids]))
        for a, b in CONTRASTS:
            if a not in table or b not in table:
                continue
            d = table[a] - table[b]
            gained, lost = int((d > 0).sum()), int((d < 0).sum())
            p = mcnemar(gained, lost)
            res["contrast"][f"{a}-{b}"] = {"value": float(d.mean()), "ci": interval(d[boots].mean(axis=1)),
                                           "gained": gained, "lost": lost, "p": p}
            if (a, b) == ("self", "uk"):
                pvalues[m] = p
        if "en" in table:
            gap = table["en"].mean() - table["uk"].mean()
            for c in ("self", "ext"):
                gain = table[c].mean() - table["uk"].mean()
                res["contrast"][f"{c}-uk"]["recovered"] = float(gain / gap) if gap else None
        out[m] = res
    return out, pvalues


def by_group(models: list[str], data: dict, items: dict[str, Item]) -> dict:
    groups = {}
    for i, it in items.items():
        if it.source == "gmmlu":
            groups.setdefault(it.category, []).append(i)
            label = {"CS": "culturally sensitive", "CA": "culturally agnostic"}.get(it.cultural, "not annotated")
            groups.setdefault(label, []).append(i)
    out = {}
    for g, ids in sorted(groups.items()):
        out[g] = {"n": len(ids)}
        for c in ("uk", "en", "self", "ext"):
            out[g][c] = float(np.mean([data["correct"][(m, c)][i] for m in models for i in ids]))
    return out


def translation_form(rows: list[dict], items: dict[str, Item]) -> dict:
    acc = {}
    for r in rows:
        if r["kind"] == "translate" and r["id"] in items:
            acc.setdefault((r["model"], items[r["id"]].source), []).append(prompts.well_formed(r["text"]))
    return {f"{m}|{s}": float(np.mean(v)) for (m, s), v in acc.items()}


def figure(summary: dict, models: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    marks = {"uk": ("o", "black", "black", "UK"), "en": ("s", "white", "black", "EN"),
             "self": ("^", "0.6", "0.6", "SELF"), "ext": ("D", "white", "0.35", "EXT")}
    for ax, source in zip(axes, SOURCES, strict=True):
        res = summary["sources"][source]
        y = np.arange(len(models))[::-1]
        for c, (marker, face, edge, label) in marks.items():
            if c not in res[models[0]]["accuracy"]:
                continue
            x = [res[m]["accuracy"][c]["value"] * 100 for m in models]
            ax.scatter(x, y, marker=marker, facecolors=face, edgecolors=edge, s=46, linewidths=1.2, label=label,
                       zorder=3)
        ax.set_yticks(y, [LABELS[m] for m in models])
        ax.set_xlabel("Accuracy, %")
        ax.set_title(SOURCES[source], fontsize=10)
        ax.grid(axis="x", color="0.85")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def check_complete(rows: list[dict], items: dict[str, Item]) -> None:
    """The Holm family covers all models and sources, so every planned answer must be present."""
    have = {(r["model"], r["condition"], r["id"]) for r in rows if r["kind"] == "answer"}
    expected = {(m, c, i) for m in MODELS for i, it in items.items()
                for c in ("uk", "en", "self", "ext") if c != "en" or it.question_en is not None}
    missing = expected - have
    if missing:
        raise ValueError(f"{len(missing)} planned answers are missing, e.g. {sorted(missing)[0]}")


def sensitivity(name: str, ids: list[str], data: dict, items: dict[str, Item], pvalues: dict) -> dict:
    """ZNO analysis on a subset of items, with the Holm family completed by the planned Global-MMLU tests."""
    result, p = analyse_source("zno", MODELS, data, items, ids)
    return {"items": name, "n": len(ids), "models": result,
            "holm": holm(pvalues | {f"zno|{m}": v for m, v in p.items()})}


def main(test: list[Item], results: Path) -> None:
    items = {it.id: it for it in test}
    rows = load_rows(results / "calls_test.jsonl")
    check_complete(rows, items)
    data = outcomes(rows, items)
    summary = {"sources": {}, "holm": {}}
    pvalues = {}
    for source in SOURCES:
        summary["sources"][source], p = analyse_source(source, MODELS, data, items)
        pvalues.update({f"{source}|{m}": v for m, v in p.items()})
    summary["holm"] = holm(pvalues)
    with_image = [i for i in first_occurrences(items, "zno") if not lost_image(items[i])]
    summary["zno_sensitivity"] = [
        sensitivity("first occurrence of each repeated question", first_occurrences(items, "zno"), data, items,
                    pvalues),
        sensitivity("also without questions whose picture is missing", with_image, data, items, pvalues),
    ]
    summary["groups"] = by_group(MODELS, data, items)
    summary["tokens"] = tokens(rows, items)
    summary["translation_form"] = translation_form(rows, items)
    summary["external_translator"] = EXTERNAL_TRANSLATOR
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figures = results.parent / "figures"
    figures.mkdir(exist_ok=True)
    figure(summary, MODELS, figures / "accuracy.png")
