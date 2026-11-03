from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


CONFIG_PATH = Path("config/thresholds.yaml")
FALLBACK_DB_PATH = Path("photo_jig_vision.db")
TABLE_NAME = "prediction_runs"

BASE_COLUMNS = [
    "run_id",
    "run_timestamp",
    "analysis_id",
    "cell",
    "face",
    "image_num",
    "image_name",
    "customer",
    "model_version",
    "background_ratio",
    "nonactive_ratio",
    "li_salt_ratio",
    "mean_confidence",
    "low_confidence_ratio",
    "high_conf_bg_ratio",
    "drift_score_before",
    "drift_score_after",
    "correction_applied",
    "normalization_skipped_reason",
    "retraining_candidate",
    "qc_warning",
    "csv_path",
    "mask_path",
    "pass_image_path",
]


st.set_page_config(page_title="Photo Jig Vision Monitoring", layout="wide")


def load_config(config_path: Path) -> dict:
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def resolve_config_db_path(config_path: Path) -> tuple[Path, str]:
    cfg = load_config(config_path)
    storage_cfg = cfg.get("storage", {}) or {}
    sqlite_cfg = storage_cfg.get("sqlite", {}) or {}
    db_name = (
        sqlite_cfg.get("path")
        or storage_cfg.get("sqlite_path")
        or cfg.get("sqlite_path")
        or str(FALLBACK_DB_PATH)
    )
    shared_path = (
        sqlite_cfg.get("shared_path")
        or storage_cfg.get("shared_path")
        or cfg.get("sqlite_shared_path")
    )

    app_root = config_path.parent.parent if config_path.parent.name == "config" else Path.cwd()
    configured_path = Path(db_name).expanduser()
    local_path = configured_path if configured_path.is_absolute() else app_root / configured_path

    if shared_path:
        shared_db_path = Path(shared_path).expanduser() / configured_path.name
        if shared_db_path.exists():
            return shared_db_path, "shared_path"
        return local_path, "local fallback"

    return local_path, "configured path"


def readonly_connection(db_path: Path) -> sqlite3.Connection:
    try:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as exc:
        if "invalid uri authority" not in str(exc).lower():
            raise

    # SQLite URI mode=ro rejects Windows UNC authorities such as file://skba/...
    # For UNC paths, open the existing file normally and force the connection
    # into query-only mode so the dashboard remains read-only.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    return conn


def table_columns(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    except sqlite3.Error:
        return []
    return [str(row[1]) for row in rows]


@st.cache_data(show_spinner=False)
def load_prediction_runs(db_path_text: str, mtime_ns: int) -> tuple[pd.DataFrame, list[str], str | None]:
    db_path = Path(db_path_text)
    try:
        with readonly_connection(db_path) as conn:
            available = table_columns(conn)
            if not available:
                return pd.DataFrame(columns=BASE_COLUMNS), [], f"Table not found: {TABLE_NAME}"

            select_columns = [column for column in BASE_COLUMNS if column in available]
            if not select_columns:
                return pd.DataFrame(columns=BASE_COLUMNS), available, "No known monitoring columns found."

            query = f"SELECT {', '.join(select_columns)} FROM {TABLE_NAME}"
            frame = pd.read_sql_query(query, conn)
    except Exception as exc:
        return pd.DataFrame(columns=BASE_COLUMNS), [], str(exc)

    for column in BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["run_timestamp_dt"] = pd.to_datetime(
        frame["run_timestamp"],
        errors="coerce",
    )
    numeric_columns = [
        "image_num",
        "background_ratio",
        "nonactive_ratio",
        "li_salt_ratio",
        "mean_confidence",
        "low_confidence_ratio",
        "high_conf_bg_ratio",
        "drift_score_before",
        "drift_score_after",
        "correction_applied",
        "retraining_candidate",
        "qc_warning",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame, available, None


def db_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def has_values(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and frame[column].notna().any()


def metric_value(value, default: str = "-") -> str:
    if pd.isna(value):
        return default
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def trend_source(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    available = [
        column
        for column in value_columns
        if column in frame.columns and frame[column].notna().any()
    ]
    if not available or frame.empty:
        return pd.DataFrame()

    plot_frame = frame.copy()
    if plot_frame["run_timestamp_dt"].notna().any():
        plot_frame = plot_frame.dropna(subset=["run_timestamp_dt"])
        if plot_frame.empty:
            return pd.DataFrame()
        plot_frame["date"] = plot_frame["run_timestamp_dt"].dt.floor("D")
        grouped = plot_frame.groupby("date", as_index=False)[available].mean(numeric_only=True)
        return grouped.melt(id_vars="date", value_vars=available, var_name="metric", value_name="value")

    plot_frame["row"] = range(1, len(plot_frame) + 1)
    return plot_frame.melt(id_vars="row", value_vars=available, var_name="metric", value_name="value")


def line_chart(frame: pd.DataFrame, value_columns: list[str], title: str) -> None:
    data = trend_source(frame, value_columns)
    if data.empty:
        st.info(f"No data available for {title}.")
        return

    x_column = "date" if "date" in data.columns else "row"
    fig = px.line(data, x=x_column, y="value", color="metric", markers=True, title=title)
    fig.update_layout(legend_title_text="", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def breakdown_chart(frame: pd.DataFrame, column: str, title: str) -> None:
    if not has_values(frame, column):
        st.info(f"No data available for {title}.")
        return
    counts = (
        frame[column]
        .fillna("Unknown")
        .replace("", "Unknown")
        .value_counts()
        .reset_index()
    )
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def apply_sidebar_filters(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()

    if has_values(filtered, "customer"):
        customers = sorted(str(value) for value in filtered["customer"].dropna().unique() if str(value))
        selected = st.sidebar.multiselect("Customer", customers)
        if selected:
            filtered = filtered[filtered["customer"].astype(str).isin(selected)]

    if has_values(filtered, "model_version"):
        versions = sorted(str(value) for value in filtered["model_version"].dropna().unique() if str(value))
        selected_versions = st.sidebar.multiselect("Model version", versions)
        if selected_versions:
            filtered = filtered[filtered["model_version"].astype(str).isin(selected_versions)]

    analysis_query = st.sidebar.text_input("Analysis ID contains", "")
    if analysis_query and "analysis_id" in filtered.columns:
        filtered = filtered[
            filtered["analysis_id"].astype(str).str.contains(analysis_query, case=False, na=False)
        ]

    if filtered["run_timestamp_dt"].notna().any():
        min_date = filtered["run_timestamp_dt"].min().date()
        max_date = filtered["run_timestamp_dt"].max().date()
        selected_range = st.sidebar.date_input("Date range", value=(min_date, max_date))
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start, end = selected_range
            filtered = filtered[
                (filtered["run_timestamp_dt"].dt.date >= start)
                & (filtered["run_timestamp_dt"].dt.date <= end)
            ]

    return filtered


st.title("Photo Jig Vision Monitoring")
st.caption("Read-only Streamlit dashboard for SQLite prediction monitoring data.")

st.sidebar.header("Configuration")
config_path_text = st.sidebar.text_input("Config path", str(CONFIG_PATH))
config_path = Path(config_path_text).expanduser()
db_path, db_source = resolve_config_db_path(config_path)
st.sidebar.caption(f"SQLite DB path from config ({db_source}):")
st.sidebar.code(str(db_path))

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()

if not db_path.exists():
    st.error(f"Database file not found: {db_path}")
    st.stop()

frame, available_columns, load_error = load_prediction_runs(str(db_path), db_mtime_ns(db_path))
if load_error:
    st.error(f"Could not read database in read-only mode: {load_error}")
    st.stop()

missing_columns = [column for column in BASE_COLUMNS if column not in available_columns]
if missing_columns:
    with st.expander("Missing DB Columns"):
        st.write(missing_columns)

filtered = apply_sidebar_filters(frame)

if filtered.empty:
    st.warning("No monitoring rows match the current filters.")
    st.stop()

st.subheader("Run Summary")
metric_cols = st.columns(5)
metric_cols[0].metric("Total processed images", f"{len(filtered):,}")
metric_cols[1].metric(
    "Total AnalysisIDs",
    f"{filtered['analysis_id'].nunique(dropna=True):,}" if "analysis_id" in filtered else "-",
)
if filtered["run_timestamp_dt"].notna().any():
    date_range = f"{filtered['run_timestamp_dt'].min():%Y-%m-%d} to {filtered['run_timestamp_dt'].max():%Y-%m-%d}"
else:
    date_range = "-"
metric_cols[2].metric("Date range", date_range)
metric_cols[3].metric("Avg Li-salt ratio", metric_value(filtered["li_salt_ratio"].mean()))
metric_cols[4].metric(
    "Retraining candidates",
    f"{int(filtered['retraining_candidate'].fillna(0).sum()):,}",
)

left, right = st.columns(2)
with left:
    breakdown_chart(filtered, "customer", "Customer Breakdown")
with right:
    breakdown_chart(filtered, "model_version", "Model Version Breakdown")

st.subheader("Output Trends")
line_chart(filtered, ["li_salt_ratio"], "Li-salt Ratio Trend")
line_chart(
    filtered,
    ["nonactive_ratio", "background_ratio"],
    "Nonactive / Background Ratio Trend",
)

st.subheader("Confidence Trends")
line_chart(filtered, ["mean_confidence"], "Mean Confidence Trend")
line_chart(filtered, ["low_confidence_ratio"], "Low-confidence Ratio Trend")

st.subheader("Drift Trends")
line_chart(filtered, ["drift_score_before"], "Drift Score Before Trend")

st.subheader("Top Drift Images")
top_drift_columns = [
    "run_timestamp",
    "analysis_id",
    "cell",
    "face",
    "image_num",
    "image_name",
    "customer",
    "model_version",
    "drift_score_before",
    "drift_score_after",
    "mean_confidence",
    "low_confidence_ratio",
    "li_salt_ratio",
    "mask_path",
    "pass_image_path",
]
top_drift = filtered.dropna(subset=["drift_score_before"]).sort_values(
    "drift_score_before",
    ascending=False,
)
if top_drift.empty:
    st.info("No drift scores available.")
else:
    st.dataframe(top_drift[top_drift_columns].head(50), use_container_width=True)

st.subheader("Retraining Candidates")
candidate_columns = [
    "run_timestamp",
    "analysis_id",
    "cell",
    "face",
    "image_num",
    "image_name",
    "customer",
    "model_version",
    "drift_score_before",
    "drift_score_after",
    "mean_confidence",
    "low_confidence_ratio",
    "li_salt_ratio",
    "normalization_skipped_reason",
    "mask_path",
    "pass_image_path",
]
candidates = filtered[filtered["retraining_candidate"].fillna(0).astype(int) == 1]
if candidates.empty:
    st.info("No retraining candidates in the current filter.")
else:
    st.dataframe(candidates[candidate_columns], use_container_width=True)
