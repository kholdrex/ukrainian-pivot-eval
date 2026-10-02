"""Prompts for answering and for translating a question with its options, and parsing of the replies."""

import re

from .data import LETTERS

ANSWER_SYSTEM = (
    "Answer the multiple-choice question. Reply with the letter of the correct option (A, B, C or D) only."
)

TRANSLATE_SYSTEM = (
    "Translate the multiple-choice question below from Ukrainian into English. Keep the layout: one line starting "
    "with 'Question:' followed by the options A, B, C and D in the same order. Do not answer the question and do "
    "not add anything else."
)

HOMOGLYPHS = str.maketrans("АВС", "ABC")
LEADING_RE = re.compile(r"^\W*([ABCD])(?![A-Za-z])")
STATED_RE = re.compile(r"(?:answer|відповідь)(?: is)?\W*([ABCD])(?![A-Za-z])", re.IGNORECASE)
OPTION_RE = re.compile(r"^\s*([ABCD])[.)][ \t]*(\S?)", re.MULTILINE)
QUESTION_RE = re.compile(r"Question:\s*\S(?:.|\n)*?^\s*A[.)]", re.MULTILINE)


def block(question: str, options: tuple[str, ...]) -> str:
    lines = [f"Question: {question.strip()}"] + [f"{LETTERS[k]}. {o.strip()}" for k, o in enumerate(options)]
    return "\n".join(lines)


def answer_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": f"{text}\nAnswer:"}]


def translate_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": TRANSLATE_SYSTEM}, {"role": "user", "content": text}]


def parse_answer(reply: str) -> int | None:
    """Option index from a reply; Cyrillic letters that look like A, B, C are read as the Latin ones."""
    reply = reply.strip().translate(HOMOGLYPHS)
    match = LEADING_RE.match(reply) or STATED_RE.search(reply)
    return LETTERS.index(match.group(1).upper()) if match else None


def well_formed(translation: str) -> bool:
    """A translation keeps a non-empty question and exactly the options A-D, in order and not empty."""
    options = OPTION_RE.findall(translation)
    return (
        QUESTION_RE.search(translation) is not None
        and [letter for letter, _ in options] == list(LETTERS)
        and all(start for _, start in options)
    )
