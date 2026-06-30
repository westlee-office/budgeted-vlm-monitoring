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


def write_dataset_tracks(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begingroup\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.11\\linewidth}>{\\raggedright\\arraybackslash}p{0.13\\linewidth}>{\\raggedright\\arraybackslash}p{0.16\\linewidth}>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.24\\linewidth}}",
        "\\toprule",
        "Track & Source & Scale & Streams & Role \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['track']} & {row['source']} & {row['source_scale']} & {row['episode_streams']} & {row['role']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\endgroup"])
    (TABLES / "dataset_tracks.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_setup(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{p{0.22\\linewidth}p{0.68\\linewidth}}",
        "\\toprule",
        "Component & Setting \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row['component']} & {row['setting']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "experiment_setup.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_by_dataset(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & CLIP Recall & \\method{} Recall & \\method{} TTD (s) & \\method{} FA/h \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {tex_float(row['clip_recall'] * 100, 1)} & "
            f"{tex_float(row['triage_recall'] * 100, 1)} & {tex_float(row['triage_ttd'], 1)} & "
            f"{tex_float(row['triage_fa_h'], 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "by_dataset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def write_failure_modes(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begingroup\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.16\\linewidth}r>{\\raggedright\\arraybackslash}p{0.30\\linewidth}>{\\raggedright\\arraybackslash}p{0.27\\linewidth}}",
        "\\toprule",
        "Failure mode & Share & Symptom & Next fix \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['failure']} & {tex_float(row['share'] * 100, 0)}\\% & {row['symptom']} & {row['next_fix']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\endgroup"])
    (TABLES / "failure_modes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def write_stream_scaling(rows: List[Dict[str, Any]]) -> None:
    uniform = " ".join(f"({row['streams']},{row['uniform'] * 100})" for row in rows)
    clip = " ".join(f"({row['streams']},{row['clip_topk'] * 100})" for row in rows)
    ours = " ".join(f"({row['streams']},{row['triagevlm'] * 100})" for row in rows)
    dense_cost = " ".join(f"({row['streams']},{row['dense_gpu_s_h']})" for row in rows)
    text = r"""\begin{tikzpicture}
\begin{axis}[
  width=0.72\linewidth,
  height=0.43\linewidth,
  xlabel={Number of streams},
  ylabel={Event Recall@B (\%)},
  xmin=24, xmax=272,
  ymin=30, ymax=90,
  xtick={32,64,128,256},
  grid=both,
  legend style={at={(0.03,0.03)},anchor=south west,font=\scriptsize},
]
\addplot+[mark=square*, thick] coordinates {""" + uniform + r"""};
\addlegendentry{Uniform}
\addplot+[mark=triangle*, thick] coordinates {""" + clip + r"""};
\addlegendentry{CLIP top-k}
\addplot+[mark=star, mark size=3pt, very thick] coordinates {""" + ours + r"""};
\addlegendentry{\method}
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "stream_scaling_tikz.tex").write_text(text, encoding="utf-8")

    cost_text = r"""\begin{tikzpicture}
\begin{axis}[
  width=0.55\linewidth,
  height=0.32\linewidth,
  xlabel={Number of streams},
  ylabel={Dense GPU-s/h},
  xtick={32,64,128,256},
  ybar,
  ymin=0,
  grid=major,
]
\addplot+[fill=gray!25] coordinates {""" + dense_cost + r"""};
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "dense_cost_scaling_tikz.tex").write_text(cost_text, encoding="utf-8")


def _timeline_marks(values: List[int], y: float, color: str, label: str) -> List[str]:
    lines = []
    for value in values:
        lines.append(f"\\draw[{color}, thick] ({value},{y - 0.18}) -- ({value},{y + 0.18});")
    if values:
        lines.append(f"\\node[anchor=west, font=\\scriptsize, {color}] at ({max(values) + 2},{y}) {{{label}}};")
    return lines


def write_timeline(data: Dict[str, Any]) -> None:
    lines = [
        "\\begin{tikzpicture}[x=0.045cm,y=0.75cm]",
        "\\draw[->] (0,0) -- (160,0) node[right, font=\\scriptsize] {time (s)};",
        "\\foreach \\x in {0,40,80,120,160} {\\draw (\\x,0.08) -- (\\x,-0.08) node[below, font=\\scriptsize] {\\x};}",
    ]
    y = 3.0
    for stream in data["streams"]:
        lines.append(f"\\node[anchor=east, font=\\scriptsize] at (-3,{y}) {{{stream['stream']} ({stream['label']})}};")
        lines.append(f"\\draw[gray!50] (0,{y}) -- (160,{y});")
        if stream.get("event_start") is not None:
            lines.append(
                f"\\draw[red!25, line width=5pt] ({stream['event_start']},{y}) -- ({stream['event_end']},{y});"
            )
            lines.append(f"\\node[font=\\scriptsize, red!70!black] at ({(stream['event_start'] + stream['event_end']) / 2},{y + 0.35}) {{event}};")
        lines.extend(_timeline_marks(stream.get("anomaly_queries", []), y - 0.12, "blue!70!black", "anomaly top-k"))
        lines.extend(_timeline_marks(stream.get("triage_queries", []), y + 0.12, "green!45!black", "\\method"))
        if stream.get("triage_detection") is not None:
            lines.append(f"\\node[star, star points=5, fill=orange!80, draw=orange!80, minimum size=4pt, inner sep=0pt] at ({stream['triage_detection']},{y + 0.45}) {{}};")
        y -= 1.0
    lines.extend(["\\end{tikzpicture}"])
    (FIGURES / "timeline_tikz.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_assumed()
    write_main_results(data["main_results"])
    write_dataset_tracks(data["dataset_tracks"])
    write_experiment_setup(data["experiment_setup"])
    write_by_dataset(data["by_dataset_results"])
    write_ablation(data["ablation"])
    write_failure_modes(data["failure_modes"])
    write_benchmark_comparison(data["benchmark_comparison"])
    write_architecture()
    write_frontier(data["main_results"])
    write_ttd_cdf(data["ttd_cdf"])
    write_stream_scaling(data["stream_scaling"])
    write_timeline(data["timeline"])
    print(f"Wrote assets under {PAPER}")


if __name__ == "__main__":
    main()
