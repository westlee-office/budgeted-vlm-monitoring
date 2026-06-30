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


def short_method(name: str) -> str:
    mapping = {
        "Random scheduler": "Random",
        "Uniform round-robin": "Uniform",
        "Motion top-k": "Motion",
        "Anomaly-score top-k": "Anomaly",
        "CLIP prompt top-k": "CLIP",
        "Dense VLM every stream": "Dense",
        "TriageVLM": "\\method{}",
    }
    return mapping.get(name, name)


def point_at(points: List[List[float]], x_value: float) -> float:
    for x, y in points:
        if x == x_value:
            return y
    raise ValueError(f"Missing point at x={x_value}")


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
  node distance=0.72cm and 0.64cm,
  box/.style={draw, rounded corners=2pt, align=center, inner sep=4pt, minimum height=0.68cm, font=\scriptsize},
  stream/.style={box, fill=gray!10, minimum width=1.75cm},
  module/.style={box, fill=blue!7, minimum width=2.05cm},
  decision/.style={box, fill=orange!12, minimum width=2.05cm},
  output/.style={box, fill=green!9, minimum width=1.85cm},
  arrow/.style={-Latex, thick}
]
\node[stream] (streams) {Stream bank\\$N$ cameras};
\node[module, right=of streams] (cheap) {Cheap scan\\motion, CLIP, anomaly};
\node[decision, right=of cheap] (policy) {VOI scorer\\rank streams};
\node[module, right=of policy] (vlm) {VLM verifier\\selected clips};
\node[output, right=of vlm] (report) {Alert +\\summary};
\node[module, below=0.62cm of policy] (memory) {Memory state\\uncertainty, cooldown};

\draw[arrow] (streams) -- (cheap);
\draw[arrow] (cheap) -- (policy);
\draw[arrow] (policy) -- (vlm);
\draw[arrow] (vlm) -- (report);
\draw[arrow] (vlm.south) |- (memory.east);
\draw[arrow] (memory.north) -- (policy.south);
\draw[arrow] (cheap.south) |- (memory.west);
\end{tikzpicture}
"""
    (FIGURES / "architecture_tikz.tex").write_text(text, encoding="utf-8")


def write_frontier(rows: List[Dict[str, Any]]) -> None:
    sparse = [row for row in rows if row["method"] != "Dense VLM every stream" and not row.get("ours")]
    dense = [row for row in rows if row["method"] == "Dense VLM every stream"][0]
    ours = [row for row in rows if row.get("ours")][0]
    sparse_coords = " ".join(f"({row['gpu_s_h']},{row['recall'] * 100})" for row in sparse)
    dense_coord = f"({dense['gpu_s_h']},{dense['recall'] * 100})"
    ours_coord = f"({ours['gpu_s_h']},{ours['recall'] * 100})"
    text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.66\linewidth,
  xlabel={GPU-seconds per hour},
  ylabel={Event Recall@B (\%)},
  xmode=log,
  xmin=800,
  xmax=9000,
  ymin=35,
  ymax=90,
  xtick={1000,2000,4000,8000},
  xticklabels={1k,2k,4k,8k},
  grid=both,
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  legend style={at={(0.97,0.03)},anchor=south east,font=\scriptsize,draw=none,fill=white},
]
\addplot+[only marks, mark=*, mark size=1.8pt, blue!65!black] coordinates {""" + sparse_coords + r"""};
\addlegendentry{sparse baselines}
\addplot+[only marks, mark=star, mark size=4pt, red!70!black, thick] coordinates {""" + ours_coord + r"""};
\addlegendentry{\method}
\addplot+[only marks, mark=diamond*, mark size=2.4pt, black!65] coordinates {""" + dense_coord + r"""};
\addlegendentry{dense VLM}
\node[anchor=west, font=\scriptsize, red!70!black] at (axis cs:1210,""" + tex_float(ours["recall"] * 100, 1) + r""") {\method};
\node[anchor=west, font=\scriptsize, black!65] at (axis cs:5050,""" + tex_float(dense["recall"] * 100, 1) + r""") {dense};
\node[anchor=south west, font=\scriptsize, blue!65!black, align=left] at (axis cs:1120,54) {equal-call\\baselines};
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "frontier_tikz.tex").write_text(text, encoding="utf-8")


def write_recall_gain(rows: List[Dict[str, Any]]) -> None:
    ours = [row for row in rows if row.get("ours")][0]
    baseline_order = [
        "CLIP prompt top-k",
        "Anomaly-score top-k",
        "Motion top-k",
        "Uniform round-robin",
        "Random scheduler",
    ]
    coords = []
    labels = []
    for method in baseline_order:
        row = [candidate for candidate in rows if candidate["method"] == method][0]
        labels.append(short_method(method))
        coords.append(f"({tex_float((ours['recall'] - row['recall']) * 100, 1)},{short_method(method)})")
    text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.66\linewidth,
  xbar,
  xmin=0,
  xmax=46,
  xlabel={Recall gain over baseline (points)},
  symbolic y coords={""" + ",".join(labels) + r"""},
  ytick=data,
  y dir=reverse,
  nodes near coords,
  nodes near coords style={font=\scriptsize},
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  grid=major,
]
\addplot+[fill=red!55, draw=red!70!black] coordinates {""" + " ".join(coords) + r"""};
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "recall_gain_tikz.tex").write_text(text, encoding="utf-8")


def write_fa_recall(rows: List[Dict[str, Any]]) -> None:
    sparse = [row for row in rows if row["method"] != "Dense VLM every stream" and not row.get("ours")]
    dense = [row for row in rows if row["method"] == "Dense VLM every stream"][0]
    ours = [row for row in rows if row.get("ours")][0]
    sparse_coords = " ".join(f"({row['false_alarms_h']},{row['recall'] * 100})" for row in sparse)
    text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.66\linewidth,
  xlabel={False alarms per hour},
  ylabel={Event Recall@B (\%)},
  xmin=3.8,
  xmax=10.0,
  ymin=35,
  ymax=90,
  grid=both,
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
]
\addplot+[only marks, mark=*, mark size=1.8pt, blue!65!black] coordinates {""" + sparse_coords + r"""};
\addplot+[only marks, mark=star, mark size=4pt, red!70!black, thick] coordinates {(""" + tex_float(ours["false_alarms_h"], 2) + "," + tex_float(ours["recall"] * 100, 1) + r""")};
\addplot+[only marks, mark=diamond*, mark size=2.4pt, black!65] coordinates {(""" + tex_float(dense["false_alarms_h"], 2) + "," + tex_float(dense["recall"] * 100, 1) + r""")};
\node[anchor=east, font=\scriptsize, blue!65!black] at (axis cs:9.65,68) {sparse baselines};
\node[anchor=west, font=\scriptsize, red!70!black] at (axis cs:""" + tex_float(ours["false_alarms_h"] + 0.15, 2) + "," + tex_float(ours["recall"] * 100, 1) + r""") {\method};
\node[anchor=south west, font=\scriptsize, black!65] at (axis cs:""" + tex_float(dense["false_alarms_h"] + 0.12, 2) + "," + tex_float(dense["recall"] * 100, 1) + r""") {dense};
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "fa_recall_tikz.tex").write_text(text, encoding="utf-8")


def write_ttd_cdf(curves: List[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  width=\\linewidth,",
        "  height=0.66\\linewidth,",
        "  xlabel={Detection delay (s)},",
        "  ylabel={Fraction of detected events},",
        "  xmin=0, xmax=120, ymin=0, ymax=1.02,",
        "  grid=both,",
        "  tick label style={font=\\scriptsize},",
        "  label style={font=\\scriptsize},",
        "  legend style={at={(0.97,0.03)},anchor=south east,font=\\scriptsize,draw=none,fill=white},",
        "]",
        "\\addplot+[black!55, dashed, no marks, forget plot] coordinates {(30,0) (30,1)};",
        "\\node[anchor=south west, font=\\scriptsize, black!65] at (axis cs:31,0.03) {30s};",
    ]
    for curve in curves:
        coords = " ".join(f"({x},{y})" for x, y in curve["points"])
        style = "very thick" if curve.get("ours") else "thick"
        lines.append(f"\\addplot+[{style}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{curve['method']}}}")
    values_at_30 = {curve["method"]: point_at(curve["points"], 30) for curve in curves}
    lines.append(
        "\\node[anchor=west, font=\\scriptsize, fill=white, inner sep=1pt] "
        f"at (axis cs:33,{values_at_30['TriageVLM']}) {{\\method{{}} {tex_float(values_at_30['TriageVLM'] * 100, 0)}\\%}};"
    )
    lines.append(
        "\\node[anchor=west, font=\\scriptsize, fill=white, inner sep=1pt] "
        f"at (axis cs:33,{values_at_30['CLIP top-k']}) {{CLIP {tex_float(values_at_30['CLIP top-k'] * 100, 0)}\\%}};"
    )
    lines.extend(["\\end{axis}", "\\end{tikzpicture}"])
    (FIGURES / "ttd_cdf_tikz.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stream_scaling(rows: List[Dict[str, Any]]) -> None:
    uniform = " ".join(f"({row['streams']},{row['uniform'] * 100})" for row in rows)
    clip = " ".join(f"({row['streams']},{row['clip_topk'] * 100})" for row in rows)
    ours = " ".join(f"({row['streams']},{row['triagevlm'] * 100})" for row in rows)
    dense_cost = " ".join(f"({row['streams']},{tex_float(row['dense_gpu_s_h'] / 1000, 1)})" for row in rows)
    base = rows[0]
    retention_uniform = " ".join(f"({row['streams']},{row['uniform'] / base['uniform'] * 100})" for row in rows)
    retention_clip = " ".join(f"({row['streams']},{row['clip_topk'] / base['clip_topk'] * 100})" for row in rows)
    retention_ours = " ".join(f"({row['streams']},{row['triagevlm'] / base['triagevlm'] * 100})" for row in rows)
    margin = " ".join(f"({row['streams']},{(row['triagevlm'] - row['clip_topk']) * 100})" for row in rows)
    text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.66\linewidth,
  xlabel={Number of streams},
  ylabel={Event Recall@B (\%)},
  xmin=24, xmax=272,
  ymin=30, ymax=90,
  xtick={32,64,128,256},
  grid=both,
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  legend style={at={(0.03,0.03)},anchor=south west,font=\scriptsize,draw=none,fill=white},
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
  width=\linewidth,
  height=0.66\linewidth,
  xlabel={Number of streams},
  ylabel={Dense GPU-s/h (k)},
  xtick={32,64,128,256},
  ybar,
  ymin=0,
  bar width=10pt,
  nodes near coords,
  nodes near coords style={font=\scriptsize},
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  grid=major,
]
\addplot+[fill=gray!25] coordinates {""" + dense_cost + r"""};
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "dense_cost_scaling_tikz.tex").write_text(cost_text, encoding="utf-8")

    retention_text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.66\linewidth,
  xlabel={Number of streams},
  ylabel={Recall retained vs 32 streams (\%)},
  xmin=24, xmax=272,
  ymin=50, ymax=105,
  xtick={32,64,128,256},
  grid=both,
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  legend style={at={(0.03,0.03)},anchor=south west,font=\scriptsize,draw=none,fill=white},
]
\addplot+[mark=square*, thick] coordinates {""" + retention_uniform + r"""};
\addlegendentry{Uniform}
\addplot+[mark=triangle*, thick] coordinates {""" + retention_clip + r"""};
\addlegendentry{CLIP top-k}
\addplot+[mark=star, mark size=3pt, very thick] coordinates {""" + retention_ours + r"""};
\addlegendentry{\method}
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "retention_scaling_tikz.tex").write_text(retention_text, encoding="utf-8")

    margin_text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.66\linewidth,
  xlabel={Number of streams},
  ylabel={Recall margin over CLIP (points)},
  xtick={32,64,128,256},
  ybar,
  ymin=0,
  ymax=16,
  bar width=10pt,
  nodes near coords,
  nodes near coords style={font=\scriptsize},
  tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  grid=major,
]
\addplot+[fill=red!55, draw=red!70!black] coordinates {""" + margin + r"""};
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "margin_scaling_tikz.tex").write_text(margin_text, encoding="utf-8")


def _timeline_marks(values: List[int], y: float, color: str) -> List[str]:
    lines = []
    for value in values:
        lines.append(f"\\draw[{color}, line width=0.8pt] ({value},{y - 0.16}) -- ({value},{y + 0.16});")
    return lines


def write_timeline(data: Dict[str, Any]) -> None:
    lines = [
        "\\begin{tikzpicture}[x=0.050cm,y=0.62cm]",
        "\\draw[red!25, line width=4pt] (8,3.95) -- (19,3.95);",
        "\\node[anchor=west, font=\\scriptsize] at (22,3.95) {event window};",
        "\\draw[blue!70!black, line width=0.8pt] (63,3.78) -- (63,4.12);",
        "\\node[anchor=west, font=\\scriptsize] at (67,3.95) {anomaly query};",
        "\\draw[green!45!black, line width=0.8pt] (111,3.78) -- (111,4.12);",
        "\\node[anchor=west, font=\\scriptsize] at (115,3.95) {\\method{} query};",
        "\\node[star, star points=5, fill=orange!85, draw=orange!85, minimum size=4pt, inner sep=0pt] at (151,3.95) {};",
        "\\node[anchor=west, font=\\scriptsize] at (155,3.95) {first detection};",
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
        lines.extend(_timeline_marks(stream.get("anomaly_queries", []), y - 0.13, "blue!70!black"))
        lines.extend(_timeline_marks(stream.get("triage_queries", []), y + 0.13, "green!45!black"))
        if stream.get("triage_detection") is not None:
            lines.append(f"\\node[star, star points=5, fill=orange!80, draw=orange!80, minimum size=4pt, inner sep=0pt] at ({stream['triage_detection']},{y + 0.45}) {{}};")
        y -= 1.0
    lines.extend(["\\end{tikzpicture}"])
    (FIGURES / "timeline_tikz.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stream_labels = [stream["stream"] for stream in data["streams"]]
    anomaly_counts = " ".join(f"({stream['stream']},{len(stream.get('anomaly_queries', []))})" for stream in data["streams"])
    triage_counts = " ".join(f"({stream['stream']},{len(stream.get('triage_queries', []))})" for stream in data["streams"])
    xticklabels = ",".join(f"{stream['stream']} {stream['label']}" for stream in data["streams"])
    count_text = r"""\begin{tikzpicture}
\begin{axis}[
  width=\linewidth,
  height=0.62\linewidth,
  ybar,
  bar width=5pt,
  ymin=0,
  ymax=6,
  ylabel={VLM queries},
  symbolic x coords={""" + ",".join(stream_labels) + r"""},
  xtick=data,
  xticklabels={""" + xticklabels + r"""},
  x tick label style={font=\scriptsize, rotate=25, anchor=east},
  y tick label style={font=\scriptsize},
  label style={font=\scriptsize},
  legend style={at={(0.03,0.97)},anchor=north west,font=\scriptsize,draw=none,fill=white},
  grid=major,
]
\addplot+[fill=blue!45, draw=blue!70!black] coordinates {""" + anomaly_counts + r"""};
\addlegendentry{Anomaly top-k}
\addplot+[fill=green!45, draw=green!45!black] coordinates {""" + triage_counts + r"""};
\addlegendentry{\method}
\end{axis}
\end{tikzpicture}
"""
    (FIGURES / "timeline_counts_tikz.tex").write_text(count_text, encoding="utf-8")


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
    write_recall_gain(data["main_results"])
    write_fa_recall(data["main_results"])
    write_ttd_cdf(data["ttd_cdf"])
    write_stream_scaling(data["stream_scaling"])
    write_timeline(data["timeline"])
    print(f"Wrote assets under {PAPER}")


if __name__ == "__main__":
    main()
