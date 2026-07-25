"""Generate a supplementary v11+v12 comparison .docx matching the reference style.

Reads:
- Reference v11 per-class metrics from report_generation/yolo_research_final_v1.docx
  (tables 5-8 in the docx: full-set re-eval, matches methodology).
- Fresh v12 per-class metrics from each run's full_eval/per_class.json file
  (produced by pipeline.full_eval.run or colab/03_full_set_eval.ipynb).

Writes:
- analysis/reports/<ts>/v11_v12_comparison.docx with:
    * Extended Table 4 (all 8 models: Precision / Recall / F1 / mAP@0.5)
    * Per-class tables for each v12 variant (Tables 3.5-3.8 style)
    * Per-class winner across all 8 (extended Table 11 with ★ markers)

Usage:
    python analysis/build_docx_report.py \\
        --v11-docx report_generation/yolo_research_final_v1.docx \\
        --runs-dir runs \\
        --out analysis/reports
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt


CLASS_ORDER = [
    "Earmuffs or Ear plugs", "Face shield", "Footwear", "Goggles",
    "Hand gloves", "Helmet", "Masks", "Safety apron or vest", "Safety jacket",
]

V11_VARIANTS = ("yolo11n", "yolo11s", "yolo11m", "yolo11l")
V12_VARIANTS = ("yolo12n", "yolo12s", "yolo12m", "yolo12l")


def _read_v11_from_docx(docx_path: Path) -> dict:
    """Extract per-class metrics for v11 n/s/m/l from the reference docx.

    Tables 5, 6, 7, 8 in yolo_research_final_v1.docx are per-class metrics for
    v11n, v11s, v11m, v11l respectively (11 rows x 6 cols: Class/Labels/P/R/AP50/AP50-95).
    """
    d = Document(docx_path)
    # Find the four 11x6 tables in order; they map to n/s/m/l
    per_class_tables = [t for t in d.tables if len(t.rows) == 11 and len(t.columns) == 6]
    if len(per_class_tables) < 4:
        raise RuntimeError(
            f"Expected >=4 per-class (11x6) tables in {docx_path}, found {len(per_class_tables)}"
        )
    out: dict = {}
    for variant, table in zip(V11_VARIANTS, per_class_tables[:4]):
        rows = []
        overall = None
        for r_idx, row in enumerate(table.rows):
            cells = [c.text.strip() for c in row.cells]
            if r_idx == 0:  # header
                continue
            name = cells[0]
            labels = _parse_int(cells[1])
            p = _parse_float(cells[2])
            r = _parse_float(cells[3])
            ap50 = _parse_float(cells[4])
            ap50_95 = _parse_float(cells[5])
            if name.lower() == "all":
                overall = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap50_95,
                           "labels": labels}
            else:
                rows.append({"name": name, "labels": labels,
                             "precision": p, "recall": r,
                             "ap50": ap50, "ap50_95": ap50_95})
        out[variant] = {"overall": overall, "per_class": rows,
                        "source": f"{docx_path.name} (full-set re-eval)"}
    return out


def _read_v12_from_runs(runs_dir: Path) -> dict:
    """Load full_eval/per_class.json from each yolo12*_* run folder."""
    out: dict = {}
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Match run names starting with any v12 variant
        variant = next((v for v in V12_VARIANTS if sub.name.startswith(f"{v}_")), None)
        if variant is None:
            continue
        eval_path = sub / "full_eval" / "per_class.json"
        if not eval_path.exists():
            print(f"[docx] SKIP {sub.name}: no full_eval/per_class.json "
                  "(run colab/03_full_set_eval.ipynb first)")
            continue
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        overall = payload.get("overall", {})
        per_class = payload.get("per_class", [])
        # Normalize per-class list keyed by name
        rows = []
        by_name = {p["name"]: p for p in per_class}
        for cname in CLASS_ORDER:
            src = by_name.get(cname, {})
            rows.append({"name": cname,
                         "labels": None,  # not tracked in our eval payload
                         "precision": src.get("precision"),
                         "recall": src.get("recall"),
                         "ap50": src.get("ap50"),
                         "ap50_95": src.get("ap50_95")})
        out[variant] = {"overall": overall, "per_class": rows,
                        "source": sub.name + "/full_eval"}
    return out


# ---------- doc building ----------

def _fmt(v, kind="pct") -> str:
    if v is None:
        return "—"
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return "—"
    if kind == "pct":
        return f"{vf * 100:.1f}%"
    if kind == "3f":
        return f"{vf:.3f}"
    if kind == "int":
        return f"{int(vf)}"
    return str(vf)


def _add_table(doc, headers: list[str], rows: list[list[str]], title: str | None = None):
    if title:
        doc.add_paragraph(title, style="Heading 3")
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
    for r_i, row in enumerate(rows, start=1):
        cells = t.rows[r_i].cells
        for c_i, val in enumerate(row):
            cells[c_i].text = str(val)
    doc.add_paragraph("")


def _f1(p, r) -> float | None:
    if p is None or r is None:
        return None
    try:
        p, r = float(p), float(r)
    except (TypeError, ValueError):
        return None
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def build_docx(v11: dict, v12: dict, out_docx: Path) -> Path:
    doc = Document()
    doc.add_heading("YOLOv11 vs YOLOv12 — Extended Comparison", level=1)
    doc.add_paragraph(
        "Supplementary comparison tables extending the analysis in "
        "yolo_research_final_v1.docx to include the YOLOv12 family (n, s, m, l). "
        "All metrics computed on the full 1,026-image labelled set to match "
        "the methodology already reported for v11 in the original document."
    )
    doc.add_paragraph(
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}. "
        f"v11 numbers sourced from the reference docx (unchanged). "
        f"v12 numbers from full_eval/per_class.json in each run folder."
    )

    # --------------- Section 1: Overall comparison across all 8 models ---------------
    doc.add_heading("1. Overall Comparison (all 8 variants)", level=2)
    doc.add_paragraph(
        "Precision and Recall are means across the 9 classes; F1-Score is computed "
        "from those means; mAP@0.5 is the standard PASCAL-VOC metric on the full set."
    )
    rows = []
    all_data = [(v, "v11", v11.get(v)) for v in V11_VARIANTS] + \
               [(v, "v12", v12.get(v)) for v in V12_VARIANTS]
    for name, family, d in all_data:
        if d is None:
            rows.append([name.upper().replace("YOLO", "YOLOv"), "—", "—", "—", "—"])
            continue
        o = d["overall"]
        rows.append([
            name.upper().replace("YOLO", "YOLOv"),
            _fmt(o.get("precision")),
            _fmt(o.get("recall")),
            _fmt(_f1(o.get("precision"), o.get("recall"))),
            _fmt(o.get("mAP50")),
        ])
    _add_table(doc, ["Model", "Precision", "Recall", "F1-Score", "mAP@0.5"], rows)

    # --------------- Section 2: Per-class tables for each v12 variant ---------------
    doc.add_heading("2. Per-class Performance — YOLOv12 Variants", level=2)
    doc.add_paragraph(
        "Same table format as the reference docx Tables 3.1-3.4 (v11 variants). "
        "Labels denotes the number of ground-truth instances of each class."
    )
    # Labels come from v11 rows (identical across all models — same dataset)
    label_lookup = {}
    if v11.get("yolo11n"):
        for row in v11["yolo11n"]["per_class"]:
            label_lookup[row["name"]] = row.get("labels")

    for idx, variant in enumerate(V12_VARIANTS, start=5):
        d = v12.get(variant)
        title = f"Table 3.{idx}: mAP value for all classes ({variant.upper().replace('YOLO', 'YOLOv')})"
        if d is None:
            doc.add_paragraph(f"{title} — NOT YET AVAILABLE (run full-set eval)")
            continue
        pc = d["per_class"]
        rows_out = []
        # Header row: All (overall)
        o = d["overall"]
        rows_out.append([
            "All", "5 547",  # total labels from docx
            _fmt(o.get("precision"), "3f"),
            _fmt(o.get("recall"), "3f"),
            _fmt(o.get("mAP50"), "3f"),
            _fmt(o.get("mAP50_95"), "3f"),
        ])
        for r in pc:
            rows_out.append([
                r["name"],
                _fmt(label_lookup.get(r["name"]), "int") if label_lookup.get(r["name"]) else "—",
                _fmt(r.get("precision"), "3f"),
                _fmt(r.get("recall"), "3f"),
                _fmt(r.get("ap50"), "3f"),
                _fmt(r.get("ap50_95"), "3f"),
            ])
        _add_table(doc,
                   ["Class", "Labels", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
                   rows_out, title=title)

    # --------------- Section 3: Per-class winner across all 8 models ---------------
    doc.add_heading("3. Per-class mAP@0.5 across all variants (★ = best)", level=2)
    doc.add_paragraph(
        "Extends the reference docx per-class comparison to include v12. The ★ "
        "marks the highest AP@0.5 for that class across all 8 variants."
    )
    all_variants = V11_VARIANTS + V12_VARIANTS
    headers = ["Class"] + [v.upper().replace("YOLO", "YOLOv") for v in all_variants]
    rows_out = []
    for cname in CLASS_ORDER:
        row = [cname]
        vals = {}
        for v in all_variants:
            src = v11.get(v) if v.startswith("yolo11") else v12.get(v)
            if src is None:
                vals[v] = None
                continue
            match = next((p for p in src["per_class"] if p["name"] == cname), None)
            vals[v] = match.get("ap50") if match else None
        max_val = max((x for x in vals.values() if x is not None), default=None)
        for v in all_variants:
            x = vals[v]
            if x is None:
                row.append("—")
            elif max_val is not None and abs(x - max_val) < 1e-9:
                row.append(f"★ {x:.3f}")
            else:
                row.append(f"{x:.3f}")
        rows_out.append(row)
    # All classes row (overall mAP50)
    all_row = ["All classes"]
    vals = {}
    for v in all_variants:
        src = v11.get(v) if v.startswith("yolo11") else v12.get(v)
        vals[v] = src["overall"].get("mAP50") if src else None
    max_val = max((x for x in vals.values() if x is not None), default=None)
    for v in all_variants:
        x = vals[v]
        if x is None:
            all_row.append("—")
        elif max_val is not None and abs(x - max_val) < 1e-9:
            all_row.append(f"★ {x:.3f}")
        else:
            all_row.append(f"{x:.3f}")
    rows_out.append(all_row)
    _add_table(doc, headers, rows_out)

    # --------------- Data provenance appendix ---------------
    doc.add_heading("Appendix: Data Provenance", level=2)
    doc.add_paragraph("Each row above traces to one of:")
    for v in V11_VARIANTS:
        src = v11.get(v, {}).get("source", "MISSING")
        doc.add_paragraph(f"  • {v}: {src}", style="List Bullet")
    for v in V12_VARIANTS:
        src = v12.get(v, {}).get("source", "NOT YET EVALUATED")
        doc.add_paragraph(f"  • {v}: {src}", style="List Bullet")

    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_docx)
    return out_docx


def _parse_float(s: str) -> float | None:
    s = s.replace("%", "").replace(",", "").replace(" ", " ").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_int(s: str) -> int | None:
    s = s.replace(",", "").replace(" ", "").replace(" ", "").strip()
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v11-docx", type=Path,
                    default=Path("report_generation/yolo_research_final_v1.docx"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("analysis/reports"))
    args = ap.parse_args()

    if not args.v11_docx.exists():
        print(f"ERROR: v11 docx not found at {args.v11_docx}", file=sys.stderr)
        return 1
    if not args.runs_dir.is_dir():
        print(f"ERROR: runs dir not found at {args.runs_dir}", file=sys.stderr)
        return 1

    v11 = _read_v11_from_docx(args.v11_docx)
    v12 = _read_v12_from_runs(args.runs_dir)

    have_v12 = sum(1 for v in V12_VARIANTS if v in v12)
    print(f"[docx] v11 variants loaded: {len(v11)}/{len(V11_VARIANTS)}")
    print(f"[docx] v12 variants with full_eval: {have_v12}/{len(V12_VARIANTS)}")
    if have_v12 == 0:
        print("[docx] WARN: no v12 full_eval results found. Run "
              "colab/03_full_set_eval.ipynb on your v12 runs first.")

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_docx = args.out / ts / "v11_v12_comparison.docx"
    build_docx(v11, v12, out_docx)
    print(f"[docx] wrote {out_docx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
