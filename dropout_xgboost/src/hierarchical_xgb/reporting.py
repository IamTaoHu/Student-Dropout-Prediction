from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from hierarchical_xgb.data_utils import save_json


def _save_stage_report_artifacts(base_dir: Path, stage_name: str, payload: dict) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    stage_key = str(stage_name).strip().lower()
    rep_path = base_dir / f"{stage_key}_classification_report.json"
    rep_csv_path = base_dir / f"{stage_key}_classification_report.csv"
    cm_path = base_dir / f"{stage_key}_confusion_matrix.csv"
    txt_path = base_dir / f"{stage_key}_summary.txt"

    if payload.get("classification_report") is None:
        save_json({"stage": stage_name, "status": "skipped", "payload": payload}, rep_path)
        pd.DataFrame([{"status": "skipped"}]).to_csv(rep_csv_path, index=False)
        pd.DataFrame([{"status": "skipped"}]).to_csv(cm_path, index=False)
        txt_path.write_text(f"{stage_name}\nstatus: skipped\nsamples: 0\n", encoding="utf-8")
        return

    report = payload["classification_report"]
    labels = payload.get("labels", ["neg", "pos"])
    cm = payload["confusion_matrix"]

    save_json({"stage": stage_name, "metrics": payload}, rep_path)
    pd.DataFrame(report).T.to_csv(rep_csv_path, index=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, index=True)

    pos_label = labels[1] if len(labels) > 1 else "pos"
    pos_metrics = report.get(pos_label, {})
    lines = [
        f"stage: {stage_name}",
        f"samples: {payload.get('num_samples', 0)}",
        f"accuracy: {payload.get('accuracy')}",
        f"balanced_accuracy: {payload.get('balanced_accuracy')}",
        f"precision: {pos_metrics.get('precision')}",
        f"recall: {pos_metrics.get('recall')}",
        f"f1-score: {pos_metrics.get('f1-score')}",
        f"support: {pos_metrics.get('support')}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_overall_report_artifacts(
    base_dir: Path,
    labels: list[str],
    report_dict: dict,
    cm: np.ndarray,
    accuracy: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_stage1: np.ndarray | None = None,
    p_stage2: np.ndarray | None = None,
    p_stage3: np.ndarray | None = None,
    route_stage1_positive: np.ndarray | None = None,
    route_stage2_positive: np.ndarray | None = None,
    routed_to_stage3: np.ndarray | None = None,
    final_t1: float | None = None,
    final_t2: float | None = None,
    final_t3: float | None = None,
    proba4: np.ndarray | None = None,
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path = base_dir / "overall_classification_report.json"
    report_csv_path = base_dir / "overall_classification_report.csv"
    cm_path = base_dir / "overall_confusion_matrix.csv"
    txt_path = base_dir / "overall_summary.txt"
    pred_path = base_dir / "final_predictions.csv"

    save_json({"labels": labels, "classification_report": report_dict}, report_path)
    pd.DataFrame(report_dict).T.to_csv(report_csv_path, index=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, index=True)

    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})
    lines = [
        "overall_4class_metrics",
        f"accuracy: {float(accuracy):.6f}",
        f"macro_precision: {float(macro.get('precision', 0.0)):.6f}",
        f"macro_recall: {float(macro.get('recall', 0.0)):.6f}",
        f"macro_f1: {float(macro.get('f1-score', 0.0)):.6f}",
        f"weighted_f1: {float(weighted.get('f1-score', 0.0)):.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    pred_df = pd.DataFrame(
        {
            "y_true_id": y_true_arr,
            "y_pred_id": y_pred_arr,
            "y_true_label": [labels[i] for i in y_true_arr],
            "y_pred_label": [labels[i] for i in y_pred_arr],
        }
    )
    if p_stage1 is not None:
        pred_df["p_stage1"] = np.asarray(p_stage1, dtype=float)
    if p_stage2 is not None:
        pred_df["p_stage2"] = np.asarray(p_stage2, dtype=float)
    if p_stage3 is not None:
        pred_df["p_stage3"] = np.asarray(p_stage3, dtype=float)
    if route_stage1_positive is not None:
        pred_df["route_stage1_positive"] = np.asarray(route_stage1_positive, dtype=int)
    if route_stage2_positive is not None:
        pred_df["route_stage2_positive"] = np.asarray(route_stage2_positive, dtype=int)
    if routed_to_stage3 is not None:
        pred_df["routed_to_stage3"] = np.asarray(routed_to_stage3, dtype=int)
    if final_t1 is not None:
        pred_df["final_threshold_t1"] = float(final_t1)
    if final_t2 is not None:
        pred_df["final_threshold_t2"] = float(final_t2)
    if final_t3 is not None:
        pred_df["final_threshold_t3"] = float(final_t3)
    if proba4 is not None:
        p4 = np.asarray(proba4, dtype=float)
        if p4.ndim == 2 and p4.shape[1] == len(labels):
            for idx, lab in enumerate(labels):
                pred_df[f"proba_{lab}"] = p4[:, idx]
    pred_df.to_csv(pred_path, index=False)


def print_run_header(
    labels: list[str],
    label_mapping: dict[str, int],
    y: pd.Series,
    feature_mode: str,
    routing_modes: list[str],
    purpose_feature_sets: dict[str, list[str]],
    feature_counts: dict[str, int],
) -> None:
    print("Detected num_class:", len(labels))
    print("Label mapping (normalized_label -> class_id):")
    print(label_mapping)
    print("Class distribution after mapping:")
    print(y.value_counts().sort_index())
    print("Feature mode:", feature_mode)
    print("Routing modes:", routing_modes)
    print(
        f"Feature counts | distinction={len(purpose_feature_sets['distinction'])} "
        f"withdrawn={len(purpose_feature_sets['withdrawn'])} failpass={len(purpose_feature_sets['failpass'])} "
        f"union={feature_counts['union']}"
    )


def select_best_mode_row(summary_df: pd.DataFrame) -> pd.Series:
    return summary_df.sort_values(by=["test_macro_f1", "test_acc"], ascending=[False, False]).iloc[0]


def print_end_summary(
    resolved_input: Path,
    routing_modes: list[str],
    output_root: Path,
    summary_df: pd.DataFrame,
) -> None:
    best_by_acc = summary_df.sort_values(by=["test_acc", "test_macro_f1"], ascending=[False, False]).iloc[0]
    best_by_macro = select_best_mode_row(summary_df)
    print("\nRun Summary")
    print(f"Input dataset path: {resolved_input}")
    print(f"Mode names run: {', '.join(routing_modes)}")
    print(f"Output root folder: {output_root}")
    print(
        "Best mode by accuracy: "
        f"{best_by_acc['routing_mode']} "
        f"(acc={float(best_by_acc['test_acc']):.4f}, macro_f1={float(best_by_acc['test_macro_f1']):.4f})"
    )
    print(
        "Best mode by macro_f1: "
        f"{best_by_macro['routing_mode']} "
        f"(macro_f1={float(best_by_macro['test_macro_f1']):.4f}, acc={float(best_by_macro['test_acc']):.4f})"
    )


def save_summary_confusion_matrices(
    summary_dir: Path,
    per_mode_results: list[dict],
    labels: list[str],
    best_mode_name: str,
) -> dict:
    """Persist summary-level confusion matrix artifacts without recomputing predictions."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "summary_confusion_matrices.json"
    best_csv_path = summary_dir / "best_mode_confusion_matrix.csv"
    best_png_path = summary_dir / "best_mode_confusion_matrix.png"
    best_txt_path = summary_dir / "summary_best_mode.txt"

    payload_rows = []
    csv_paths: dict[str, str] = {}
    best_mode_row = None

    for result in per_mode_results:
        routing_mode = str(result["routing_mode"])
        cm = np.asarray(result["confusion_matrix"], dtype=int)
        payload_rows.append(
            {
                "routing_mode": routing_mode,
                "best_seed": int(result["best_seed"]),
                "labels": labels,
                "confusion_matrix": cm.tolist(),
                "test_acc": float(result["test_acc"]),
                "test_macro_f1": float(result["test_macro_f1"]),
                "test_bal_acc": float(result["test_bal_acc"]),
                **(
                    {"routing_score": float(result["routing_score"])}
                    if result.get("routing_score") is not None
                    else {}
                ),
                **(
                    {"min_recall": float(result["min_recall"])}
                    if result.get("min_recall") is not None
                    else {}
                ),
                **(
                    {"collapse_penalty": float(result["collapse_penalty"])}
                    if result.get("collapse_penalty") is not None
                    else {}
                ),
                **(
                    {"v2_specialist_test_f1": float(result["v2_specialist_test_f1"])}
                    if result.get("v2_specialist_test_f1") is not None
                    else {}
                ),
                **(
                    {"v2_num_pass_to_distinction_flips_test": int(result["v2_num_pass_to_distinction_flips_test"])}
                    if result.get("v2_num_pass_to_distinction_flips_test") is not None
                    else {}
                ),
            }
        )
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        mode_csv_path = summary_dir / f"confusion_matrix_{routing_mode}.csv"
        cm_df.to_csv(mode_csv_path, index=True)
        csv_paths[routing_mode] = str(mode_csv_path)
        if routing_mode == best_mode_name:
            best_mode_row = result
            cm_df.to_csv(best_csv_path, index=True)

    save_json({"labels": labels, "modes": payload_rows}, json_path)

    created = {
        "summary_confusion_json": str(json_path),
        "best_mode_confusion_csv": str(best_csv_path),
        "best_mode_confusion_png": str(best_png_path),
        "summary_best_mode_txt": str(best_txt_path),
        "per_mode_csv": csv_paths,
    }

    if best_mode_row is None:
        return created

    best_cm = np.asarray(best_mode_row["confusion_matrix"], dtype=int)
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Best Mode Confusion Matrix ({best_mode_name})")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(best_png_path, dpi=150)
    plt.close(fig)

    best_txt_path.write_text(
        "\n".join(
            [
                f"best_mode_by_macro_f1: {best_mode_name}",
                f"best_seed: {int(best_mode_row['best_seed'])}",
                f"test_acc: {float(best_mode_row['test_acc']):.6f}",
                f"test_macro_f1: {float(best_mode_row['test_macro_f1']):.6f}",
                f"test_bal_acc: {float(best_mode_row['test_bal_acc']):.6f}",
                f"selected_t1: {float(best_mode_row['selected_t1']):.6f}",
                f"selected_t2: {float(best_mode_row['selected_t2']):.6f}",
                f"selected_t3: {float(best_mode_row['selected_t3']):.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return created


def save_summary_text_report(
    summary_dir: Path,
    summary_df: pd.DataFrame,
    best_mode_row: pd.Series,
) -> Path:
    report_path = summary_dir / "summary_report.txt"
    best_by_acc = summary_df.sort_values(by=["test_acc", "test_macro_f1"], ascending=[False, False]).iloc[0]
    table_cols = [
        "routing_mode",
        "best_seed",
        "test_macro_f1",
        "test_acc",
        "test_bal_acc",
        "selected_t1",
        "selected_t2",
        "selected_t3",
    ]
    lines = [
        f"total_routing_modes_evaluated: {int(len(summary_df))}",
        f"best_mode_by_macro_f1: {best_mode_row['routing_mode']}",
        f"best_mode_by_accuracy: {best_by_acc['routing_mode']}",
        "",
        summary_df[table_cols].to_string(index=False),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_v2_summary_report(
    summary_dir: Path,
    summary_df: pd.DataFrame,
    best_mode_row: pd.Series,
    arch_version: str,
) -> Path | None:
    if arch_version != "v2":
        return None

    report_path = summary_dir / "v2_summary_report.txt"
    lines = [
        f"arch_version: {arch_version}",
        f"routing_modes_evaluated: {int(len(summary_df))}",
        f"best_mode_by_test_macro_f1: {best_mode_row['routing_mode']}",
    ]
    if "routing_score" in summary_df.columns:
        best_by_routing_score = summary_df.sort_values(
            by=["routing_score", "test_macro_f1", "test_acc"],
            ascending=[False, False, False],
        ).iloc[0]
        lines.append(f"best_mode_by_routing_score: {best_by_routing_score['routing_mode']}")

    table_cols = [
        col
        for col in [
            "routing_mode",
            "routing_score",
            "test_macro_f1",
            "test_acc",
            "test_bal_acc",
            "min_recall",
            "collapse_penalty",
            "v2_specialist_test_f1",
            "v2_num_pass_to_distinction_flips_test",
        ]
        if col in summary_df.columns
    ]
    if table_cols:
        lines.extend(["", summary_df[table_cols].to_string(index=False)])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
