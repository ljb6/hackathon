from unittest.mock import patch
from query_builder import compose_query, prompt_query


# --- compose_query ---

def test_all_fields_present():
    q = compose_query(upper_color="red", lower_color="black",
                      has_backpack=True, has_hat=False, extra="beard")
    assert "red" in q
    assert "black" in q
    assert "backpack" in q
    assert "beard" in q
    assert "hat" not in q


def test_only_upper_color():
    q = compose_query(upper_color="blue", lower_color="",
                      has_backpack=False, has_hat=False, extra="")
    assert "blue" in q
    assert "backpack" not in q


def test_only_backpack():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=True, has_hat=False, extra="")
    assert "backpack" in q


def test_only_hat():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=False, has_hat=True, extra="")
    assert "hat" in q


def test_all_empty_returns_nonempty_string():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=False, has_hat=False, extra="")
    assert isinstance(q, str) and len(q) > 0


def test_both_accessories():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=True, has_hat=True, extra="")
    assert "backpack" in q and "hat" in q


def test_extra_appended():
    q = compose_query(upper_color="", lower_color="",
                      has_backpack=False, has_hat=False, extra="tall man")
    assert "tall man" in q


def test_upper_and_lower_both_present():
    q = compose_query(upper_color="white", lower_color="blue",
                      has_backpack=False, has_hat=False, extra="")
    assert "white" in q and "blue" in q


# --- prompt_query ---

def test_prompt_query_returns_string_with_all_fields():
    with patch("builtins.input", side_effect=iter(["red", "black", "y", "y", "tall man"])):
        q = prompt_query()
    assert "red" in q and "black" in q
    assert "backpack" in q and "hat" in q
    assert "tall man" in q


def test_prompt_query_skips_empty_fields():
    with patch("builtins.input", side_effect=iter(["", "", "n", "n", ""])):
        q = prompt_query()
    assert "backpack" not in q and "hat" not in q


def test_prompt_query_prints_composed_query(capsys):
    with patch("builtins.input", side_effect=iter(["blue", "", "y", "n", ""])):
        prompt_query()
    assert "blue" in capsys.readouterr().out
