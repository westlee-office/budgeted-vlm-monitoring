#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2027"
ASSUMED = PAPER / "assumed_results.json"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


def load_assumed() -> Dict[str, Any]:
    with ASSUMED.open("r", encoding="utf-8") as f:
        return json.load(f)


def tex_float(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write_main_results(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Recall@B $\\uparrow$ & TTD (s) $\\downarrow$ & FA/h $\\downarrow$ & GPU-s/h $\\downarrow$ & VLM calls/event $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        name = row["method"]
        if row.get("ours"):
            name = "\\textbf{" + name + "}"
        lines.append(
            f"{name} & {tex_float(row['recall'] * 100, 1)} & {tex_float(row['ttd_s'], 1)} & "
            f"{tex_float(row['false_alarms_h'], 2)} & {tex_float(row['gpu_s_h'], 0)} & "
            f"{tex_float(row['calls_event'], 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Recall@B $\\uparrow$ & TTD (s) $\\downarrow$ & FA/h $\\downarrow$ & Calls/event $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']} & {tex_float(row['recall'] * 100, 1)} & {tex_float(row['ttd_s'], 1)} & "
            f"{tex_float(row['false_alarms_h'], 2)} & {tex_float(row['calls_event'], 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_comparison(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Benchmark family & Multi-stream & Online & Budgeted VLM & Event latency & Semantic reports \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["family"]] + ["\\cmark" if row[k] else "\\xmark" for k in ["multi_stream", "online", "budgeted_vlm", "event_latency", "semantic_reports"]]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "benchmark_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_architecture() -> None:
    text = r"""\begin{tikzpicture}[
  node distance=0.65cm,
  box/.style={draw, rounded corners=2pt, align=center, inner sep=5pt, minimum height=0.78cm, font=\small},
  stream/.style={box, fill=gray!10, minimum width=2.1cm},
  module/.style={box, fill=blue!7, minimum width=2.5cm},
  decision/.style={box, fill=orange!12, minimum width=2.8cm},
  output/.style={box, fill=green!9, minimum width=2.4cm},
  arrow/.style={-Latex, thick}
]
\node[stream] (streams) {Hundreds of\\video streams};
\node[module, right=of streams] (cheap) {Cheap perception\\motion / CLIP / anomaly};
\node[decision, right=of cheap] (policy) {Value-of-information\\scheduler};
\node[module, right=of policy] (vlm) {Selective\\VLM calls};
\node[module, below=of policy] (memory) {Event memory\\and uncertainty};
\node[output, right=of vlm] (report) {Incident alert\\and summary};

\draw[arrow] (streams) -- (cheap);
\draw[arrow] (cheap) -- node[above, font=\scriptsize] {stream scores} (policy);
\draw[arrow] (policy) -- node[above, font=\scriptsize] {look actions} (vlm);
\draw[arrow] (vlm) -- (report);
\draw[arrow] (vlm.south) |- (memory.east);
\draw[arrow] (memory.west) -| (policy.south);
\draw[arrow] (cheap.south) |- (memory.west);
\end{tikzpicture}
"""
    (FIGURES / "architecture_tikz.tex").write_text(text, encoding="utf-8")


def write_frontier(rows: List[Dict[str, Any]]) -> None:
    coords = []
    for row in rows:
        coords.append(f"({row['gpu_s_h']},{row['recall'] * 100})")
    ours = [row for row in rows if row.get("ours")]
    ours_coord = f"({ours[0]['gpu_s_h']},{ours[0]['recall'] * 100})" if ours else ""
    text = r"""\begin{tikzpicture}
\begin{axis}[
  width=0.72\linewidth,
  height=0.42\linewidth,
  xlabel={GPU-seconds per hour},
  ylabel={Event Recall@B (\%)},
  xmin=0,
  xmax=7600,
  ymin=35,
  ymax=90,
  grid=both,
  legend style={at={(0.03,0.97)},anchor=north west,font=\scriptsize},
]
\addplot+[mark=*, thick] coordinates {""" + " ".join(coords) + r"""};
\addlegendentry{Baselines and ours}
""" + (r"\addplot+[only marks, mark=star, mark size=3.2pt, thick] coordinates {" + ours_coord + r"};" if ours_coord else "") + r"""
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "frontier_tikz.tex").write_text(text, encoding="utf-8")


def write_ttd_cdf(curves: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  width=0.72\\linewidth,",
        "  height=0.42\\linewidth,",
        "  xlabel={Detection delay (s)},",
        "  ylabel={Fraction of detected events},",
        "  xmin=0, xmax=120, ymin=0, ymax=1.02,",
        "  grid=both,",
        "  legend style={at={(0.97,0.03)},anchor=south east,font=\\scriptsize},",
        "]",
    ]
    for curve in curves:
        coords = " ".join(f"({x},{y})" for x, y in curve["points"])
        style = "very thick" if curve.get("ours") else "thick"
        lines.append(f"\\addplot+[{style}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{curve['method']}}}")
    lines.extend(["\\end{axis}", "\\end{tikzpicture}"])
    (FIGURES / "ttd_cdf_tikz.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_assumed()
    write_main_results(data["main_results"])
    write_ablation(data["ablation"])
    write_benchmark_comparison(data["benchmark_comparison"])
    write_architecture()
    write_frontier(data["main_results"])
    write_ttd_cdf(data["ttd_cdf"])
    print(f"Wrote assets under {PAPER}")


if __name__ == "__main__":
    main()
