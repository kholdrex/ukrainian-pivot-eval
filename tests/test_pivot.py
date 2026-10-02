from pivot.analysis import holm, mcnemar
from pivot.prompts import block, parse_answer, well_formed


def test_parse_answer_variants():
    assert parse_answer("B") == 1
    assert parse_answer(" (C) because ...") == 2
    assert parse_answer("The answer is D.") == 3
    assert parse_answer("Правильна відповідь: B") == 1
    assert parse_answer("А.") == 0
    assert parse_answer("Step 1: compute the total") is None
    assert parse_answer("The answer is b.") == 1


def test_block_and_translation_form():
    text = block("Скільки буде 2 + 2?", ("3", "4", "5", "6"))
    assert text.splitlines()[0] == "Question: Скільки буде 2 + 2?"
    assert well_formed("Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6")
    assert not well_formed("Question: What is 2 + 2?\nA. 3\nB. 4\nD. 6")
    assert not well_formed("Question: What is 2 + 2?\nA. \nB. \nC. \nD.")


def test_mcnemar_exact():
    assert mcnemar(0, 0) == 1.0
    assert abs(mcnemar(10, 0) - 2 / 2**10) < 1e-12
    assert mcnemar(5, 5) == 1.0


def test_holm_is_monotone():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_lost_image_detects_empty_tables_only():
    from pivot.data import Item, lost_image

    def item(question):
        return Item("x", "zno", "History", "native", question, ("a", "b", "c", "d"), 0)

    assert lost_image(item("На фото зображено\n\n|  |  |\n| --- | --- |\n|  |  |"))
    assert not lost_image(item("Таблиця\n| a | b |\n| --- | --- |\n| 1 | 2 |"))
    assert not lost_image(item("Без таблиці"))
