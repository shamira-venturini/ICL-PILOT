from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


_REPORT_NAME = "feature_redundancy_report.md"
_ID_LIKE_NAME_HINTS = ("file", "id", "source", "relpath", "transcript", "path")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series_signature(series: pd.Series) -> tuple[str, ...]:
    return tuple(series.fillna("<NA>").astype(str).tolist())


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    value = left.corr(right, method=method)
    if pd.isna(value):
        return None
    return float(value)


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _markdown_list(items: list[str]) -> str:
    if not items:
        return "- none\n"
    return "".join(f"- `{item}`\n" for item in items)


def audit_feature_redundancy(
    input_csv: str,
    output_dir: str,
    corr_threshold: float = 0.95,
    cluster_threshold: float = 0.98,
) -> int:
    input_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    row_count = len(df)

    column_rows: list[dict[str, object]] = []
    constant_columns: list[str] = []
    id_like_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        unique_non_null = int(series.dropna().nunique())
        unique_with_null = int(series.nunique(dropna=False))
        is_constant = unique_with_null == 1
        name_hint = column.lower()
        looks_named_like_id = any(token in name_hint for token in _ID_LIKE_NAME_HINTS)
        is_text_like = not pd.api.types.is_numeric_dtype(series)
        is_id_like = unique_non_null == row_count and unique_with_null == row_count and (
            looks_named_like_id or is_text_like
        )
        if is_constant:
            constant_columns.append(column)
        if is_id_like:
            id_like_columns.append(column)

        column_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_rows": non_null,
                "missing_rows": int(series.isna().sum()),
                "unique_non_null_values": unique_non_null,
                "unique_values_including_null": unique_with_null,
                "is_constant": int(is_constant),
                "is_id_like": int(is_id_like),
            }
        )

    _write_csv(
        out_dir / "column_inventory.csv",
        [
            "column",
            "dtype",
            "non_null_rows",
            "missing_rows",
            "unique_non_null_values",
            "unique_values_including_null",
            "is_constant",
            "is_id_like",
        ],
        column_rows,
    )

    duplicate_groups: list[list[str]] = []
    by_signature: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for column in df.columns:
        by_signature[_series_signature(df[column])].append(column)
    for columns in by_signature.values():
        if len(columns) > 1:
            duplicate_groups.append(sorted(columns))
    duplicate_groups.sort(key=lambda group: (len(group), group))

    duplicate_rows: list[dict[str, object]] = []
    substantive_duplicate_groups: list[list[str]] = []
    for group in duplicate_groups:
        all_constant = all(column in constant_columns for column in group)
        if not all_constant:
            substantive_duplicate_groups.append(group)
        duplicate_rows.append(
            {
                "group_size": len(group),
                "all_constant": int(all_constant),
                "columns": " | ".join(group),
            }
        )
    _write_csv(
        out_dir / "duplicate_columns.csv",
        ["group_size", "all_constant", "columns"],
        duplicate_rows,
    )

    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    modeled_numeric_columns = [
        column
        for column in numeric_columns
        if column not in constant_columns and column not in id_like_columns
    ]

    pair_rows: list[dict[str, object]] = []
    high_corr_rows: list[dict[str, object]] = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for index, left_name in enumerate(modeled_numeric_columns):
        left_series = pd.to_numeric(df[left_name], errors="coerce")
        for right_name in modeled_numeric_columns[index + 1 :]:
            right_series = pd.to_numeric(df[right_name], errors="coerce")
            mask = left_series.notna() & right_series.notna()
            overlap = int(mask.sum())
            if overlap < 3:
                continue

            left_overlap = left_series[mask]
            right_overlap = right_series[mask]
            exact_equal = bool((left_overlap == right_overlap).all())
            pearson = _safe_corr(left_overlap, right_overlap, method="pearson")
            spearman = _safe_corr(left_overlap, right_overlap, method="spearman")
            max_abs_diff = float((left_overlap - right_overlap).abs().max())
            mean_abs_diff = float((left_overlap - right_overlap).abs().mean())

            row = {
                "left_column": left_name,
                "right_column": right_name,
                "overlap_rows": overlap,
                "pearson_r": "" if pearson is None else f"{pearson:.6f}",
                "spearman_rho": "" if spearman is None else f"{spearman:.6f}",
                "abs_pearson_r": "" if pearson is None else f"{abs(pearson):.6f}",
                "exact_equal_on_overlap": int(exact_equal),
                "max_abs_diff": f"{max_abs_diff:.6f}",
                "mean_abs_diff": f"{mean_abs_diff:.6f}",
            }
            pair_rows.append(row)

            if pearson is not None and abs(pearson) >= corr_threshold:
                high_corr_rows.append(row)
            if pearson is not None and abs(pearson) >= cluster_threshold:
                adjacency[left_name].add(right_name)
                adjacency[right_name].add(left_name)

    pair_rows.sort(
        key=lambda row: (
            -float(row["abs_pearson_r"] or 0.0),
            row["left_column"],
            row["right_column"],
        )
    )
    high_corr_rows.sort(
        key=lambda row: (
            -float(row["abs_pearson_r"] or 0.0),
            row["left_column"],
            row["right_column"],
        )
    )
    _write_csv(
        out_dir / "numeric_pair_metrics.csv",
        [
            "left_column",
            "right_column",
            "overlap_rows",
            "pearson_r",
            "spearman_rho",
            "abs_pearson_r",
            "exact_equal_on_overlap",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        pair_rows,
    )
    _write_csv(
        out_dir / "high_correlation_pairs.csv",
        [
            "left_column",
            "right_column",
            "overlap_rows",
            "pearson_r",
            "spearman_rho",
            "abs_pearson_r",
            "exact_equal_on_overlap",
            "max_abs_diff",
            "mean_abs_diff",
        ],
        high_corr_rows,
    )

    seen: set[str] = set()
    correlation_clusters: list[list[str]] = []
    for column in sorted(adjacency):
        if column in seen:
            continue
        stack = [column]
        cluster: list[str] = []
        seen.add(column)
        while stack:
            current = stack.pop()
            cluster.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(cluster) > 1:
            correlation_clusters.append(sorted(cluster))

    cluster_rows = [
        {"cluster_size": len(cluster), "columns": " | ".join(cluster)}
        for cluster in sorted(correlation_clusters, key=lambda item: (-len(item), item))
    ]
    _write_csv(out_dir / "correlation_clusters.csv", ["cluster_size", "columns"], cluster_rows)

    recommended_exclude = sorted(
        set(constant_columns)
        | set(id_like_columns)
        | {column for group in substantive_duplicate_groups for column in group[1:]}
    )

    summary = {
        "input_csv": str(input_path),
        "rows": row_count,
        "columns": len(df.columns),
        "numeric_columns": len(numeric_columns),
        "modeled_numeric_columns": len(modeled_numeric_columns),
        "constant_columns": constant_columns,
        "id_like_columns": id_like_columns,
        "duplicate_groups": duplicate_groups,
        "substantive_duplicate_groups": substantive_duplicate_groups,
        "high_correlation_pair_count": len(high_corr_rows),
        "correlation_clusters": correlation_clusters,
        "recommended_exclude_from_modeling": recommended_exclude,
        "corr_threshold": corr_threshold,
        "cluster_threshold": cluster_threshold,
    }
    (out_dir / "feature_redundancy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    top_pairs = high_corr_rows[:12]
    report_lines = [
        "# Feature Redundancy Audit\n",
        "\n",
        f"- Input: `{input_path}`\n",
        f"- Rows: `{row_count}`\n",
        f"- Columns: `{len(df.columns)}`\n",
        f"- Numeric columns considered for pairwise audit: `{len(modeled_numeric_columns)}`\n",
        f"- High-correlation threshold: `{corr_threshold}`\n",
        f"- Cluster threshold: `{cluster_threshold}`\n",
        "\n",
        "## Constant Columns\n",
        _markdown_list(constant_columns),
        "\n",
        "## ID-Like Columns\n",
        _markdown_list(id_like_columns),
        "\n",
        "## Exact Duplicate Groups\n",
    ]
    if duplicate_groups:
        for group in duplicate_groups:
            marker = "constant-only" if all(column in constant_columns for column in group) else "substantive"
            report_lines.append(f"- `{marker}`: `{', '.join(group)}`\n")
    else:
        report_lines.append("- none\n")

    report_lines.extend(
        [
            "\n",
            "## High-Correlation Clusters\n",
        ]
    )
    if correlation_clusters:
        for cluster in correlation_clusters:
            report_lines.append(f"- `{', '.join(cluster)}`\n")
    else:
        report_lines.append("- none\n")

    report_lines.extend(
        [
            "\n",
            "## Top High-Correlation Pairs\n",
        ]
    )
    if top_pairs:
        for row in top_pairs:
            report_lines.append(
                "- "
                f"`{row['left_column']}` vs `{row['right_column']}`: "
                f"pearson={_format_float(float(row['pearson_r'])) if row['pearson_r'] else 'NA'}, "
                f"spearman={_format_float(float(row['spearman_rho'])) if row['spearman_rho'] else 'NA'}, "
                f"exact_equal={row['exact_equal_on_overlap']}, "
                f"max_abs_diff={row['max_abs_diff']}\n"
            )
    else:
        report_lines.append("- none\n")

    report_lines.extend(
        [
            "\n",
            "## Recommended Exclusions Before Severity Modeling\n",
            _markdown_list(recommended_exclude),
        ]
    )
    (out_dir / _REPORT_NAME).write_text("".join(report_lines), encoding="utf-8")

    print(f"Wrote redundancy audit to {out_dir}")
    return 0
