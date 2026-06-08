"""Tests for table_report_util."""

from machine_learning_engineering.shared_libraries import table_report_util


def _sample_report(num_columns: int = 5) -> dict:
    columns = []
    for idx in range(num_columns):
        columns.append(
            {
                "name": f"col_{idx}",
                "dtype": "Float64DType",
                "null_proportion": 0.0,
                "nulls_level": "ok",
                "n_unique": idx + 1,
                "is_high_cardinality": False,
                "value_is_constant": idx == num_columns - 1,
            }
        )
    return {
        "n_rows": 100,
        "n_columns": num_columns,
        "n_constant_columns": 1,
        "columns": columns,
        "top_associations": [
            {
                "left_column_name": "col_0",
                "right_column_name": "col_1",
                "pearson_corr": 0.91,
                "cramer_v": 0.0,
            }
        ],
    }


def test_get_profile_from_state_returns_cached_profile():
    state = {"ablation_table_report_profile_1": "Dataset: 10 rows"}
    assert (
        table_report_util.get_profile_from_state(state, "1")
        == "Dataset: 10 rows"
    )


def test_get_profile_from_state_fallback():
    assert (
        table_report_util.get_profile_from_state({}, "1")
        == table_report_util.PROFILE_UNAVAILABLE
    )


def test_profile_state_key():
    assert (
        table_report_util.profile_state_key("42")
        == "ablation_table_report_profile_42"
    )


def test_format_ablation_profile_truncates_wide_tables():
    profile = table_report_util.format_ablation_profile(
        _sample_report(num_columns=40),
        max_columns=5,
    )
    assert "... and 35 more columns (see table_report.json)" in profile
    column_lines = [
        line
        for line in profile.splitlines()
        if line.startswith("- col_") and ": Float64DType" in line
    ]
    assert len(column_lines) == 5


def test_format_ablation_profile_prioritizes_non_constant_columns():
    profile = table_report_util.format_ablation_profile(
        _sample_report(num_columns=5),
        max_columns=5,
    )
    column_lines = [
        line
        for line in profile.splitlines()
        if line.startswith("- col_") and ": Float64DType" in line
    ]
    assert column_lines[-1].startswith("- col_4")


def test_format_ablation_profile_caps_associations_at_eight():
    associations = []
    for idx in range(12):
        associations.append(
            {
                "left_column_name": f"a_{idx}",
                "right_column_name": f"b_{idx}",
                "pearson_corr": 0.95 - idx * 0.01,
                "cramer_v": 0.0,
            }
        )
    report = _sample_report(num_columns=3)
    report["top_associations"] = associations
    profile = table_report_util.format_ablation_profile(report)
    association_lines = [
        line for line in profile.splitlines() if " <-> " in line
    ]
    assert len(association_lines) == 8
