# English pivot for Ukrainian questions

Code, prompts, all model replies and analysis for the paper *Does translation into English help open language models
answer Ukrainian questions?* (O. Kholodniak).

Five open models answer multiple-choice questions in four conditions:

| Condition | Text given to the model |
|---|---|
| UK | the Ukrainian question |
| EN | the English original (Global-MMLU only) |
| SELF | the model's own English translation of the Ukrainian question, answered in a new conversation |
| EXT | the English translation made by GPT-4o-mini |

The primary contrast is SELF − UK, separately for questions translated into Ukrainian (Global-MMLU) and questions
written in Ukrainian (ZNO history), with paired bootstrap intervals and exact McNemar tests with the Holm correction.

## Data

- [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), revision `0e619db`, Apache-2.0: `en` and
  `uk` test questions, 200 per subject category (seed 2026).
- [ZNO](https://huggingface.co/datasets/osyvokon/zno), revision `473616f`, MIT: all 1486 History of Ukraine questions.

Both are downloaded at the pinned revisions on first use; the selected item ids are in `data/sample.json`.

## Models

Called through OpenRouter at temperature 0 with one fixed provider each and fallbacks disabled (`experiment.py`):
Llama 3.2 3B (Parasail), Qwen 2.5 7B (Phala), Llama 3.1 8B, Gemma 3 12B and Mistral Small 3.2 24B (DeepInfra);
GPT-4o-mini (OpenAI) as the external translator.

## Layout

```
src/pivot/    data loading, prompts and reply parsing, OpenRouter client, runner, analysis
results/      calls_test.jsonl (every translation and answer with tokens, provider and time),
              calls_pilot.jsonl (30-question pilot outside the test sample), summary.json
figures/      figure of the paper
tests/        unit tests
PROTOCOL.md   analysis plan written before the test run; protocol_freeze.sha256 holds the hashes at that time
```

## Usage

Python 3.10+ and [uv](https://docs.astral.sh/uv/).

Install from the checkout (editable), because the commands read `data/` and `results/` relative to it;
`requirements.lock.txt` lists the exact package versions used.

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run pivot analyze         # results/summary.json and figures/accuracy.png from the stored replies
```

New calls need the OpenRouter key in the `OPENROUTER_API_KEY` environment variable (a full run costs a few US
dollars at 2026 prices). `uv run pivot run test` resumes `results/calls_test.jsonl`, which is already complete; to
start a fresh run, move `results/calls_test.jsonl` aside first. Hosted models can change over time, so new replies
may differ from the stored ones.

## License

Code: MIT. The stored replies contain material from Global-MMLU (Apache-2.0) and ZNO (MIT); see
`THIRD_PARTY_NOTICES.md`.
