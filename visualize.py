"""결과 시각화: Table 4 스타일 바 차트 생성"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


def load_summary(results_dir: str = "./results") -> dict:
    summary_path = Path(results_dir) / "table4_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_table4(summary: dict, save_path: str = None):
    """Table 4 스타일 수평 바 차트."""
    # 한국어 폰트 설정
    mpl.rcParams["font.family"] = "NanumGothic"
    mpl.rcParams["axes.unicode_minus"] = False

    results = summary["results"]  # 이미 accuracy 오름차순 정렬
    labels = [r["label"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#d4d4d4"] * len(results)
    # CoT-decoding 관련 전략 강조
    for i, r in enumerate(results):
        if "cot" in r["strategy"].lower():
            colors[i] = "#4CAF50"

    bars = ax.barh(labels, accuracies, color=colors, edgecolor="#333", linewidth=0.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Table 4 | Decoding Strategy Comparison\n"
        f"Model: {summary['model']}  |  Dataset: {summary['dataset']}  |  "
        f"N={summary['num_samples']}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(accuracies) * 1.15)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    summary = load_summary(results_dir)
    plot_table4(summary, save_path=str(Path(results_dir) / "table4_chart.png"))
