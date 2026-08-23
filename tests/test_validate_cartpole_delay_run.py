import pytest

from scripts.validate_cartpole_delay_run import csv_integral


@pytest.mark.parametrize(
    ("serialized", "expected"),
    [("0", 0), ("0.0", 0), ("64.0", 64), ("20000", 20_000)],
)
def test_csv_integral_accepts_integer_valued_serializations(serialized, expected):
  assert csv_integral(serialized) == expected


@pytest.mark.parametrize("serialized", ["2.5", "nan", "inf"])
def test_csv_integral_rejects_nonintegral_or_nonfinite_values(serialized):
  with pytest.raises(ValueError, match="integer-valued"):
    csv_integral(serialized)
