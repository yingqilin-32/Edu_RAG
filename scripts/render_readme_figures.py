"""Render the README comparison chart from saved reports; no API or DB calls.

Run in a documentation environment with ReportLab installed. It is intentionally
not an application dependency. Output: docs/assets/evaluation-highlights.svg.
"""

import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/evaluation-highlights.svg"
INK, MUTED = HexColor("#172b4d"), HexColor("#52647c")
BLUE, TEAL, PALE = HexColor("#2563eb"), HexColor("#07837b"), HexColor("#9aaac0")


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def label(drawing, x, y, text, *, size=12, color=INK, anchor="start", bold=False):
    drawing.add(String(x, y, text, fontName="Helvetica-Bold" if bold else "Helvetica",
                       fontSize=size, fillColor=color, textAnchor=anchor))


def panel(drawing, *, x, title, subtitle, gain, categories, values, labels, maximum, step, colors):
    drawing.add(Rect(x, 71, 366, 337, rx=12, ry=12, fillColor=HexColor("#ffffff"),
                     strokeColor=HexColor("#dce4ef"), strokeWidth=0.8))
    label(drawing, x + 22, 377, title, size=17, bold=True)
    label(drawing, x + 22, 354, subtitle, size=11, color=MUTED)
    label(drawing, x + 22, 325, gain, size=15, color=TEAL, bold=True)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = x + 53, 125, 284, 159
    chart.data = [values]
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 10
    chart.categoryAxis.labels.fillColor = MUTED
    chart.categoryAxis.strokeColor = HexColor("#c1ccdb")
    chart.categoryAxis.tickDown = 0
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, maximum, step
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 9
    chart.valueAxis.labels.fillColor = MUTED
    chart.valueAxis.visibleGrid = True
    chart.valueAxis.gridStrokeColor = HexColor("#e9edf4")
    chart.valueAxis.strokeColor = HexColor("#c1ccdb")
    chart.valueAxis.tickLeft = 0
    chart.barSpacing, chart.groupSpacing = 12, 25
    chart.bars.strokeColor = None
    for index, color in enumerate(colors):
        chart.bars[0, index].fillColor = color
    drawing.add(chart)
    # Center labels on the single series' category slots. Leave headroom above
    # every bar, and keep the value axis at zero in all three comparisons.
    for index, (value, text) in enumerate(zip(values, labels, strict=True)):
        center = chart.x + chart.width * (index + 0.5) / len(values)
        label(drawing, center, chart.y + chart.height * value / maximum + 9,
              text, size=12, bold=True, anchor="middle")


def main():
    retrieval = read("data/v3/four_way/report_deepseek.json")["summary"]["retrieval"]
    policies = read("data/v5/runs/v2/report.json")["metrics"]
    memory = read("data/v6/runs/v2/analysis.json")["modes"]
    raw, built = (retrieval[name]["ndcg_at_5"] for name in ("raw_bm25", "query_bm25"))
    fixed, adaptive = (policies[name]["nominal_model_calls"] for name in ("always_retry", "adaptive"))
    assert retrieval["raw_bm25"]["queries"] == retrieval["query_bm25"]["queries"] == 36
    assert policies["always_retry"]["partitions"]["new_authored_complex"]["completed"] == 18
    assert policies["adaptive"]["partitions"]["new_authored_complex"]["completed"] == 18
    assert all(memory[mode]["positive_followups"] == 14 for mode in ("none", "history", "structured"))

    drawing = Drawing(1200, 505)
    drawing.add(Rect(0, 0, 1200, 505, fillColor=HexColor("#f4f7fc"), strokeColor=None))
    label(drawing, 32, 466, "Measured changes, with explicit baselines", size=24, bold=True)
    label(drawing, 32, 439, "Edu RAG  |  Paired development experiments  |  Recorded 19 October 2026",
          size=12, color=MUTED)
    panel(drawing, x=32, title="V3.2 / Query construction", subtitle="nDCG@5, 36 retrieval queries; higher is better",
          gain=f"+{(built / raw - 1) * 100:.2f}% relative ranking score",
          categories=["Raw BM25", "Constructed BM25"], values=[raw, built],
          labels=[f"{raw:.4f}", f"{built:.4f}"], maximum=1.0, step=0.2, colors=[PALE, BLUE])
    panel(drawing, x=417, title="V5 / Adaptive retry", subtitle="Model calls across 92 cases; lower is better",
          gain=f"{(1 - adaptive / fixed) * 100:.1f}% fewer model calls",
          categories=["Fixed retries", "Adaptive"], values=[fixed, adaptive],
          labels=[str(fixed), str(adaptive)], maximum=400, step=100, colors=[PALE, TEAL])
    panel(drawing, x=802, title="V6 / Session memory", subtitle="Strict successful follow-ups, out of 14",
          gain="+11 successes with full history",
          categories=["None", "History", "Structured"],
          values=[memory[m]["strict_followup_success"] for m in ("none", "history", "structured")],
          labels=[f'{memory[m]["strict_followup_success"]}/14' for m in ("none", "history", "structured")],
          maximum=14, step=2, colors=[PALE, BLUE, TEAL])
    label(drawing, 600, 88, "V5: both policies complete 18/24 complex tasks.", size=10,
          color=MUTED, anchor="middle")
    drawing.add(Line(32, 57, 1168, 57, strokeColor=HexColor("#dce4ef"), strokeWidth=0.8))
    label(drawing, 32, 36, "Panels use different datasets and units. Ranking and task success are not production answer accuracy.",
          size=11, color=MUTED)
    label(drawing, 32, 18, "Source reports: data/v3/four_way/  |  data/v5/runs/v2/  |  data/v6/runs/v2/  (full methods and costs in README)",
          size=10, color=MUTED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(drawing, str(OUTPUT))
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
