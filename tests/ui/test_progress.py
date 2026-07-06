import sys
from io import StringIO

from miplib.ui.progress import ProgressBar


def test_default_construction():
    bar = ProgressBar()
    assert bar.min == 0
    assert bar.max == 100
    assert bar.span == 100
    assert bar.amount == 0


def test_str_after_construction():
    bar = ProgressBar(total_width=20)
    result = str(bar)
    assert len(result) == 20
    assert "0%" in result


def test_update_amount_zero_percent():
    bar = ProgressBar(total_width=20)
    bar.update_amount(0)
    result = str(bar)
    assert ">" in result
    assert "0%" in result


def test_update_amount_full():
    bar = ProgressBar(total_width=40)
    bar.update_amount(100)
    result = str(bar)
    assert "100%" in result


def test_update_amount_half():
    bar = ProgressBar(max_value=200, total_width=40)
    bar.update_amount(100)
    result = str(bar)
    assert "50%" in result


def test_update_amount_clamps_to_min():
    bar = ProgressBar(min_value=10, max_value=100)
    bar.update_amount(5)
    assert bar.amount == 10


def test_update_amount_clamps_to_max():
    bar = ProgressBar(max_value=100)
    bar.update_amount(150)
    assert bar.amount == 100


def test_show_percentage_true():
    bar = ProgressBar(show_percentage=True, total_width=20)
    bar.update_amount(50)
    assert "%" in str(bar)


def test_show_percentage_false_shows_count():
    bar = ProgressBar(show_percentage=False, max_value=200, total_width=40)
    bar.update_amount(75)
    assert "75/200" in str(bar)


def test_update_comment():
    bar = ProgressBar()
    bar.update_comment(" hello ")
    bar.update_amount(50)
    assert bar._comment == " hello "


def test_call_writes_to_stdout():
    bar = ProgressBar(total_width=20)
    captured = StringIO()
    sys.stdout = captured
    bar(50)
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert "50%" in output
    assert "\r" in output


def test_update_amount_changes_bar():
    bar = ProgressBar(total_width=30)
    first = str(bar)
    bar.update_amount(50)
    second = str(bar)
    assert first != second


def test_call_writes_after_comment_change():
    bar = ProgressBar(total_width=20)
    bar(50)
    bar.update_comment("x")
    captured = StringIO()
    sys.stdout = captured
    bar(50)
    sys.stdout = sys.__stdout__
    assert "x" in captured.getvalue()


def test_eta_appears_after_multiple_updates():
    bar = ProgressBar(total_width=40)
    bar.update_amount(10)
    result1 = str(bar)
    assert "ETA:" in result1
    bar.update_amount(20)
    result2 = str(bar)
    assert "ETA:" in result2


def test_zero_span_runs():
    bar = ProgressBar(min_value=5, max_value=5)
    bar.update_amount(5)
    assert bar.amount == 5


def test_prefix_in_output():
    bar = ProgressBar(prefix="Test: ", total_width=30)
    captured = StringIO()
    sys.stdout = captured
    bar(100)
    sys.stdout = sys.__stdout__
    assert captured.getvalue().startswith("\r Test: ")


def test_custom_dims():
    bar = ProgressBar(min_value=10, max_value=110, total_width=50)
    bar.update_amount(60)
    result = str(bar)
    assert "50%" in result
    assert len(result) > 50  # includes ETA


def test_bar_structure_empty():
    bar = ProgressBar(total_width=10)
    bar.update_amount(0)
    result = str(bar)
    assert ">" in result
    assert "0%" in result


def test_bar_structure_full():
    bar = ProgressBar(total_width=10)
    bar.update_amount(100)
    result = str(bar)
    assert ">" not in result
    assert "100%" in result
