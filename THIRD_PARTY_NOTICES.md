# Third-party data

`results/calls_test.jsonl` and `results/calls_pilot.jsonl` contain the questions and options of the items used, in
the models' translations and replies, and `data/sample.json` lists their ids.

- Global-MMLU (https://huggingface.co/datasets/CohereLabs/Global-MMLU, revision 0e619db), published by Cohere Labs
  under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). It is derived from MMLU
  (https://github.com/hendrycks/test), MIT licence, Copyright (c) 2020 Dan Hendrycks (`LICENSES/MIT-MMLU.txt`).
  The stored questions are unmodified or translated by the models listed in README.md.
- ZNO (https://huggingface.co/datasets/osyvokon/zno, revision 473616f): questions of Ukraine's external independent
  testing collected from https://zno.osvita.ua. The dataset card declares the MIT licence; the dataset provides no
  licence file or copyright line.

The MIT licence of this repository covers the code.
