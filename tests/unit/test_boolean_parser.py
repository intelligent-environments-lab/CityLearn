import pytest

from citylearn.utilities import parse_bool


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("FALSE", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
    ],
)
def test_parse_bool_accepts_supported_inputs(value, expected):
    assert parse_bool(value, path="test.value") is expected


def test_parse_bool_uses_default_for_none():
    assert parse_bool(None, default="false", path="test.value") is False
    assert parse_bool(None, default=1, path="test.value") is True


def test_parse_bool_rejects_invalid_tokens():
    with pytest.raises(ValueError, match="test.value"):
        parse_bool("maybe", path="test.value")

    with pytest.raises(ValueError, match="test.value"):
        parse_bool(2, path="test.value")
