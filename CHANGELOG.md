# Changes after the analysis plan was fixed

The plan (`PROTOCOL.md`) was written after a 30-question pilot and before the test run on 2026-10-02. The stored
replies were not changed. Changes to the code afterwards:

- The runner retries calls that failed after all attempts in further passes instead of stopping, and runs 32
  requests in parallel instead of 12. This affects only the order in which calls were made.
- The figure was redrawn in greyscale with the legend above the panels.
- The OpenRouter key is read from the `OPENROUTER_API_KEY` environment variable instead of a fixed file path.
- After review: cultural-sensitivity labels of Global-MMLU are kept as CS, CA or not annotated (the first version
  merged CS and CA); the letter after "answer"/"відповідь" may be lowercase (no stored reply was affected); two
  post-hoc sensitivity analyses of ZNO keep each repeated question once and, in addition, drop the three questions
  whose picture is missing from the dataset (`zno_sensitivity` in `summary.json`).
- The translation-form check requires a "Question:" line with a non-empty question and non-empty options A-D; the
  first version checked only the option letters. With the current check 779 of the 2,686 Llama 3.2 3B translations
  fail: 421 have an empty question, 253 keep the question and options but drop the "Question:" label, 25 contain
  only an answer letter, and the rest change the options or the layout in other ways.
- Analysis requires the complete set of planned answers, so that the Holm correction always covers the planned
  family. Jobs are named tuples; the model list moved to `experiment.py`; figures are written to
  `figures/accuracy.png`.
- The repository has no history from before the freeze, so `protocol_freeze.sha256` documents the hashes at that
  time but the frozen files themselves are not included.
- Figures use Times New Roman.
