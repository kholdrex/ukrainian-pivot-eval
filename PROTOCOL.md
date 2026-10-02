# Analysis plan

Fixed on 2026-10-02 after a 30-question pilot on Global-MMLU items outside the test sample, before any test call.

## Question
For small and mid-sized open language models, does translating a Ukrainian multiple-choice question into English
before answering improve accuracy, and how much of the gap to the original English question does it recover?

## Items
- Global-MMLU (revision 0e619db), parallel `en`/`uk` test questions: 200 per subject category (6 categories,
  1200 items), seed 2026. The Ukrainian questions are translations of the English originals.
- ZNO, History of Ukraine (revision 473616f), all 1486 train and test questions: native Ukrainian exam questions
  without an English version.

## Models and conditions
Llama 3.2 3B, Qwen 2.5 7B, Llama 3.1 8B, Gemma 3 12B, Mistral Small 3.2 24B through OpenRouter with a fixed
provider each; temperature 0. Conditions: UK (Ukrainian question), EN (English original; Global-MMLU only),
SELF (the same model first translates the question and options into English, then answers the translation in a new
conversation), EXT (translation by GPT-4o-mini, answered by the model). The answering instruction is identical in
all conditions. Option letters are Latin A–D everywhere.

## Outcome
An answer is correct when the first option letter in the reply (or the letter after "answer"/"відповідь") is the
gold option. Replies without a letter count as incorrect; their rate is reported per condition.

## Analyses
1. Primary: SELF − UK accuracy per model and source, 95% paired bootstrap interval over items (2000 resamples),
   exact McNemar test with Holm correction over the 10 model × source contrasts.
2. EN − UK (translation gap of the benchmark) and EXT − UK; share of the EN − UK gap recovered by SELF and EXT.
3. Transitions UK→SELF: wrong→right and right→wrong counts.
4. By Global-MMLU category, and culturally sensitive vs agnostic items.
5. Cost: prompt and completion tokens per question for each condition.
6. Translation form: share of translations that keep the question line and options A–D.
