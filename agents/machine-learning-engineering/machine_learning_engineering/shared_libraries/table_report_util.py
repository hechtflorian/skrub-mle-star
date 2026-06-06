"""Utilities for building ablation data profiles from skrub TableReport."""

import json
import re

import pandas as pd
import skrub

_TARGET_PATTERNS = (
    re.compile(r"""target_col\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
    re.compile(r"""TARGET\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
)


def extract_target_from_code(code: str) -> str | None:
    """Extract target column name from generated solution code."""
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(code)
        if match:
            return match.group(1)
    return None


def load_table_report_dict(train_csv_path: str) -> dict:
    """Load train data and return TableReport JSON as a dictionary."""
    train_df = pd.read_csv(train_csv_path)
    report = skrub.TableReport(
        train_df,
        verbose=False,
        plot_distributions=False,
    )
    return json.loads(report.json())


def _is_categorical_dtype(dtype: str) -> bool:
    dtype_lower = dtype.lower()
    return any(
        token in dtype_lower
        for token in ("string", "categorical", "object", "category")
    )


def _is_numeric_dtype(dtype: str) -> bool:
    dtype_lower = dtype.lower()
    return any(
        token in dtype_lower for token in ("float", "int", "double", "decimal")
    )


def format_ablation_profile(report: dict, *, target_col: str | None = None) -> str:
    """Format TableReport JSON into a compact ablation-oriented summary."""
    lines: list[str] = []
    n_rows = report.get("n_rows", "?")
    n_columns = report.get("n_columns", "?")
    n_constant = report.get("n_constant_columns", 0)
    lines.append(
        f"Dataset: {n_rows} rows, {n_columns} columns ({n_constant} constant)"
    )
    if target_col:
        lines.append(f"Target (from current solution): {target_col}")

    columns = report.get("columns", [])
    numeric_count = 0
    categorical_count = 0
    missing_columns: list[str] = []

    lines.append("")
    lines.append("Columns:")
    for column in columns:
        name = column.get("name", "?")
        dtype = column.get("dtype", "?")
        null_proportion = column.get("null_proportion", 0.0) or 0.0
        nulls_level = column.get("nulls_level", "?")
        n_unique = column.get("n_unique", "?")
        high_cardinality = column.get("is_high_cardinality", False)

        flags: list[str] = []
        if column.get("value_is_constant"):
            flags.append("constant")
        if column.get("is_duration"):
            flags.append("duration")
        flag_suffix = f", flags={','.join(flags)}" if flags else ""

        lines.append(
            f"- {name}: {dtype}, nulls={null_proportion:.1%} ({nulls_level}), "
            f"unique={n_unique}, high_card={high_cardinality}{flag_suffix}"
        )

        if _is_numeric_dtype(dtype):
            numeric_count += 1
        elif _is_categorical_dtype(dtype):
            categorical_count += 1

        if null_proportion > 0 or nulls_level not in ("ok", None):
            missing_columns.append(name)

    lines.append("")
    lines.append(
        "Numeric: "
        f"{numeric_count} | String/Categorical: {categorical_count} | "
        f"Missing columns: {len(missing_columns)}"
    )

    associations = report.get("top_associations", [])
    filtered_associations: list[tuple[float, float, dict]] = []
    for association in associations:
        pearson_abs = abs(association.get("pearson_corr") or 0.0)
        cramer_v = association.get("cramer_v") or 0.0
        if pearson_abs >= 0.85 or cramer_v >= 0.5:
            filtered_associations.append((pearson_abs, cramer_v, association))
    filtered_associations.sort(
        key=lambda item: max(item[0], item[1]),
        reverse=True,
    )
    filtered_associations = filtered_associations[:8]

    if filtered_associations:
        lines.append("")
        lines.append("Top associations:")
        for _, cramer_v, association in filtered_associations:
            left = association.get("left_column_name", "?")
            right = association.get("right_column_name", "?")
            pearson = association.get("pearson_corr")
            if pearson is not None:
                lines.append(f"- {left} <-> {right}: pearson={pearson:.2f}")
            else:
                lines.append(f"- {left} <-> {right}: cramer_v={cramer_v:.2f}")

    return "\n".join(lines)
