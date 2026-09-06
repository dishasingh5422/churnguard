import pytest

from src.feature_encoding import FEATURE_ENCODINGS, encode_category


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("gender", "Male", 1),
        ("MultipleLines", "No phone service", 1),
        ("MultipleLines", "Yes", 2),
        ("InternetService", "DSL", 0),
        ("InternetService", "Fiber optic", 1),
        ("InternetService", "No", 2),
        ("OnlineSecurity", "No", 0),
        ("OnlineSecurity", "No internet service", 1),
        ("OnlineSecurity", "Yes", 2),
        ("Contract", "Two year", 2),
        ("PaymentMethod", "Electronic check", 2),
    ],
)
def test_category_encoding(feature, value, expected):
    assert encode_category(feature, value) == expected


def test_categories_do_not_collapse_to_same_number():
    for mapping in FEATURE_ENCODINGS.values():
        assert len(mapping.values()) == len(set(mapping.values()))


def test_unknown_category_raises_clear_error():
    with pytest.raises(ValueError):
        encode_category("InternetService", "Satellite")
