from pathlib import Path

import numpy as np
import pandas as pd

import test_dashboard


def sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filename": "a.png",
                "model_name": "m1",
                "li_salt_percent": 10.0,
                "background_percent": 80.0,
                "nonactive_percent": 5.0,
                "edge_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 100.0,
            },
            {
                "filename": "a.png",
                "model_name": "m2",
                "li_salt_percent": 20.0,
                "background_percent": 70.0,
                "nonactive_percent": 5.0,
                "edge_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 100.0,
            },
            {
                "filename": "b.png",
                "model_name": "m1",
                "li_salt_percent": 12.0,
                "background_percent": 78.0,
                "nonactive_percent": 5.0,
                "edge_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 100.0,
            },
            {
                "filename": "b.png",
                "model_name": "m2",
                "li_salt_percent": 14.0,
                "background_percent": 76.0,
                "nonactive_percent": 5.0,
                "edge_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 100.0,
            },
        ]
    )


def test_variability_stats_calculates_range_and_std() -> None:
    stats = test_dashboard.variability_stats(sample_results())
    row = stats[
        (stats["filename"] == "a.png")
        & (stats["class_name"] == "Lithium Salt")
    ].iloc[0]

    assert row["mean"] == 15.0
    assert row["range"] == 10.0
    assert row["max_absolute_difference"] == 10.0


def test_pairwise_model_differences_averages_absolute_class_diffs() -> None:
    matrix = test_dashboard.pairwise_model_differences(sample_results(), "Lithium Salt")

    assert matrix.loc["m1", "m2"] == 6.0
    assert matrix.loc["m2", "m1"] == 6.0


def test_model_reliability_scores_ranks_lower_deviation_first() -> None:
    frame = pd.DataFrame(
        [
            {
                "filename": "a.png",
                "model_name": "stable",
                "li_salt_percent": 10.0,
                "background_percent": 80.0,
                "nonactive_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 100.0,
            },
            {
                "filename": "a.png",
                "model_name": "outlier",
                "li_salt_percent": 30.0,
                "background_percent": 60.0,
                "nonactive_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 95.0,
            },
            {
                "filename": "b.png",
                "model_name": "stable",
                "li_salt_percent": 10.0,
                "background_percent": 80.0,
                "nonactive_percent": 5.0,
                "original_exists": True,
                "mask_exists": True,
                "percent_sum": 100.0,
            },
            {
                "filename": "b.png",
                "model_name": "outlier",
                "li_salt_percent": 35.0,
                "background_percent": 55.0,
                "nonactive_percent": 5.0,
                "original_exists": True,
                "mask_exists": False,
                "percent_sum": 95.0,
            },
        ]
    )
    scores = test_dashboard.model_reliability_scores(frame)

    assert scores.iloc[0]["model_name"] == "stable"
    assert scores.iloc[0]["reliability_score"] >= scores.iloc[1]["reliability_score"]


def test_model_card_flags_marks_lithium_extremes_and_warning() -> None:
    frame = pd.DataFrame(
        [
            {"model_name": "low", "li_salt_percent": 5.0},
            {"model_name": "middle", "li_salt_percent": 10.0},
            {"model_name": "high", "li_salt_percent": 30.0},
        ]
    )

    flags = test_dashboard.model_card_flags(frame, threshold=8.0)

    assert flags["low"]["lowest_li"] is True
    assert flags["high"]["highest_li"] is True
    assert flags["middle"]["highest_li"] is False
    assert flags["high"]["variability_warning"] is True
    assert flags["middle"]["variability_warning"] is False


def test_enhanced_source_csv_matrix_uses_only_enhanced_csvs(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    model_a = models_root / "model_a"
    model_b = models_root / "nested" / "model_b"
    model_a.mkdir(parents=True)
    model_b.mkdir(parents=True)
    (model_a / "source_csvs.txt").write_text(
        "\n".join(
            [
                "data/labeled/a/enhanced_cell_1.csv",
                "data/labeled/a/basic_cell_1.csv",
                "data/labeled/a/enhanced_cell_2.csv",
            ]
        ),
        encoding="utf-8",
    )
    (model_b / "source_csvs.txt").write_text(
        "\n".join(
            [
                "data\\labeled\\b\\enhanced_cell_2.csv",
                "data\\labeled\\b\\raw_cell_3.csv",
            ]
        ),
        encoding="utf-8",
    )

    signature = test_dashboard.model_source_csv_signature(["model_a", "model_b"], models_root)
    matrix, missing = test_dashboard.load_enhanced_source_csv_matrix(signature)

    assert missing == []
    assert matrix["Enhanced CSV"].tolist() == [
        "data/labeled/a/enhanced_cell_1.csv",
        "data/labeled/a/enhanced_cell_2.csv",
        "data/labeled/b/enhanced_cell_2.csv",
    ]
    assert matrix["model_a"].tolist() == ["●", "●", ""]
    assert matrix["model_b"].tolist() == ["", "", "●"]


def test_build_consensus_returns_agreement_metrics() -> None:
    mask_a = np.array([[1, 0], [2, 2]], dtype=np.int16)
    mask_b = np.array([[1, 0], [2, 1]], dtype=np.int16)

    result = test_dashboard.build_consensus([("m1", mask_a), ("m2", mask_b)])

    assert result is not None
    assert result["agreement_ratio"] == 0.75
    assert result["disagreement_ratio"] == 0.25
    assert result["consensus"].shape == (2, 2)


def test_discover_result_dirs_uses_existing_test_model_structure(tmp_path: Path) -> None:
    valid = tmp_path / "model_a"
    invalid = tmp_path / "model_b"
    valid.mkdir()
    invalid.mkdir()
    (valid / "test_results.csv").write_text("filename\n", encoding="utf-8")

    assert test_dashboard.discover_result_dirs(tmp_path) == [valid]


def test_attach_review_status_defaults_pending() -> None:
    results = pd.DataFrame(
        [
            {"model_name": "m1", "filename": "a.png"},
            {"model_name": "m2", "filename": "a.png"},
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "model_name": "m1",
                "filename": "a.png",
                "review_status": "Pass",
                "reviewed_at": "now",
            }
        ]
    )

    merged = test_dashboard.attach_review_status(results, reviews)

    assert merged["review_status"].tolist() == ["Pass", "Pending"]


def test_model_summary_table_counts_manual_pass_fail() -> None:
    frame = pd.DataFrame(
        [
            {"model_name": "m1", "filename": "a.png", "review_status": "Pass"},
            {"model_name": "m1", "filename": "b.png", "review_status": "Fail"},
            {"model_name": "m2", "filename": "a.png", "review_status": "Pass"},
            {"model_name": "m2", "filename": "b.png", "review_status": "Pending"},
        ]
    )

    summary = test_dashboard.model_summary_table(
        frame,
        ["m1", "m2"],
        {"m1": ["data/a.csv", "data/b.csv"], "m2": ["data/c.csv"]},
    )

    assert summary.loc[summary["Model Name"] == "m1", "#Pass"].iloc[0] == 1
    assert summary.loc[summary["Model Name"] == "m1", "#Fail"].iloc[0] == 1
    assert summary.loc[summary["Model Name"] == "m1", "Pass Rate"].iloc[0] == "50.0%"
    assert summary.loc[summary["Model Name"] == "m2", "#Pass"].iloc[0] == 2
    assert summary.loc[summary["Model Name"] == "m2", "Train Dataset"].iloc[0] == "c.csv"


def test_dataset_influence_table_compares_pass_and_fail_models() -> None:
    datasets = {
        "good": ["shared.csv", "good_only.csv"],
        "bad": ["shared.csv", "bad_only.csv"],
    }

    table = test_dashboard.dataset_influence_table(["good"], ["bad"], datasets)

    good_only = table[table["Dataset"] == "good_only.csv"].iloc[0]
    bad_only = table[table["Dataset"] == "bad_only.csv"].iloc[0]
    assert good_only["Pass Advantage"] == 1.0
    assert bad_only["Fail Advantage"] == 1.0


def test_add_review_border_uses_status_color() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    bordered = test_dashboard.add_review_border(image, "Fail", border_px=1)

    assert bordered.shape == (4, 5, 3)
    assert bordered[0, 0].tolist() == list(test_dashboard.FAIL_BORDER_COLOR)
    assert bordered[1, 1].tolist() == [0, 0, 0]


def test_add_review_border_leaves_pass_unchanged() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    bordered = test_dashboard.add_review_border(image, "Pass", border_px=1)

    assert bordered.shape == image.shape
