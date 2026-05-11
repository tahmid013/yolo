"""Scan a runs/ directory and emit a Markdown comparison report + plots.

Usage:
    python analysis/build_report.py --runs-dir runs --out analysis/reports
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# Allow running as a script
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.collect import RunRecord, load_runs
    from analysis.plots import (
        copy_confusion_matrices, per_class_ap, size_vs_map, training_curves,
    )
    from analysis.tables import build_comparison, to_markdown
else:
    from .collect import RunRecord, load_runs
    from .plots import (
        copy_confusion_matrices, per_class_ap, size_vs_map, training_curves,
    )
    from .tables import build_comparison, to_markdown


def _utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _detect_warnings(records: list[RunRecord]) -> list[str]:
    msgs: list[str] = []
    hashes = {r.meta.get("config_hash") for r in records if r.meta.get("config_hash")}
    if len(hashes) > 1:
        msgs.append(
            "Runs use **different config hashes** — hyperparameters are not identical "
            "across runs. Comparisons reflect both model and HP differences."
        )
    shas = {
        (r.meta.get("dataset") or {}).get("sha256")
        for r in records
        if (r.meta.get("dataset") or {}).get("sha256")
    }
    if len(shas) > 1:
        msgs.append(
            "Runs were trained on **different dataset versions** (sha256 mismatch). "
            "Test-set comparisons are NOT apples-to-apples."
        )
    missing_test = [r.name for r in records if not r.per_class]
    if missing_test:
        msgs.append(
            "Runs missing test-split evaluation: " + ", ".join(missing_test)
            + ". Re-run pipeline.evaluate.run() on these and re-build the report."
        )
    return msgs


def _run_appendix(record: RunRecord, out_root: Path) -> str:
    train = record.meta.get("train") or {}
    env = record.meta.get("env") or {}
    dataset = record.meta.get("dataset") or {}
    bits = [
        f"### {record.name}",
        "",
        f"- model: `{record.meta.get('model')}`",
        f"- config: `{record.meta.get('config_path')}` (hash `{record.meta.get('config_hash')}`)",
        f"- dataset: version `{dataset.get('version')}`, sha256 `{(dataset.get('sha256') or '')[:16]}`",
        f"- env: ultralytics `{env.get('ultralytics')}`, torch `{env.get('torch')}`, GPU `{env.get('gpu_name')}`",
        f"- wall time: {train.get('wall_time_sec')} s ({train.get('epochs_completed')}/{train.get('epochs_requested')} epochs"
        + (", early-stopped)" if train.get("early_stopped") else ")"),
    ]
    if record.per_class and record.per_class.get("per_class"):
        bits += ["", "| class | AP50 | AP50-95 | P | R |", "|---|---:|---:|---:|---:|"]
        for c in sorted(record.per_class["per_class"], key=lambda x: x["id"]):
            def f(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
            bits.append(
                f"| {c['name']} | {f(c.get('ap50'))} | {f(c.get('ap50_95'))} | "
                f"{f(c.get('precision'))} | {f(c.get('recall'))} |"
            )
    return "\n".join(bits)


def build_report(runs_dir: Path, out: Path) -> Path:
    ts = _utc_ts()
    report_dir = out / ts
    plots_dir = report_dir / "plots"
    cm_dir = plots_dir / "confusion_matrices"
    report_dir.mkdir(parents=True, exist_ok=True)

    records = load_runs(runs_dir)
    print(f"[report] loaded {len(records)} runs from {runs_dir}")

    if not records:
        (report_dir / "report.md").write_text(
            f"# YOLO runs report — {ts}\n\n_No runs found under `{runs_dir}`._\n",
            encoding="utf-8",
        )
        return report_dir

    df = build_comparison(records)
    df.to_csv(report_dir / "comparison.csv", index=False)
    table_md = to_markdown(df)

    p_curves = training_curves(records, plots_dir / "training_curves.png")
    p_perclass = per_class_ap(records, plots_dir / "per_class_ap.png")
    p_size = size_vs_map(records, plots_dir / "size_vs_map.png")
    p_cms = copy_confusion_matrices(records, cm_dir)

    warnings = _detect_warnings(records)

    md_lines: list[str] = [
        f"# YOLO runs report — {ts}",
        "",
        f"Runs source: `{runs_dir}` ({len(records)} run(s))",
        "",
    ]
    if warnings:
        md_lines += ["## ⚠ Caveats", ""]
        for w in warnings:
            md_lines.append(f"- {w}")
        md_lines.append("")
    md_lines += ["## Comparison", "", table_md, ""]

    if p_curves:
        md_lines += ["## Training curves", "", f"![training_curves]({p_curves.relative_to(report_dir).as_posix()})", ""]
    if p_perclass:
        md_lines += ["## Per-class AP@0.5 (test split)", "", f"![per_class_ap]({p_perclass.relative_to(report_dir).as_posix()})", ""]
    if p_size:
        md_lines += ["## Model size vs accuracy", "", f"![size_vs_map]({p_size.relative_to(report_dir).as_posix()})", ""]
    if p_cms:
        md_lines += ["## Confusion matrices (test split)", ""]
        for cm in p_cms:
            md_lines.append(f"### {cm.stem}\n\n![cm]({cm.relative_to(report_dir).as_posix()})\n")

    md_lines += ["## Per-run details", ""]
    for r in records:
        md_lines.append(_run_appendix(r, report_dir))
        md_lines.append("")

    (report_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[report] wrote {report_dir / 'report.md'}")
    return report_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("analysis/reports"))
    args = ap.parse_args()
    build_report(args.runs_dir.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
