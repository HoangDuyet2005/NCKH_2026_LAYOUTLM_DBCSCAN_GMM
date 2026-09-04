"""
Vẽ biểu đồ trực quan hoá kết quả 5-Fold Cross-Validation (đọc từ data/kfold_results.json)
để minh hoạ cho mục "5-Fold Cross-Validation" trong README.

Gồm 2 biểu đồ:
  1. F1-Score của từng fold (bar chart) + đường trung bình ± std, so với F1 của
     lần chia đơn ban đầu (Step 5) -- minh hoạ độ ổn định qua nhiều lần chia.
  2. F1-Score theo từng loại thực thể: Baseline rule-based vs. LayoutLMv3 (1 lần
     chia, 9 mẫu) vs. LayoutLMv3 (5-Fold out-of-fold, 45 mẫu) -- minh hoạ vì sao
     số liệu đo trên tập nhỏ có thể gây hiểu lầm.

Chạy: python src/plot_kfold_results.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Số liệu per-entity (%) -- nguồn: step5b_eval_breakdown.py (baseline, 1 lần chia)
# và step5c_kfold_cv.py (out-of-fold). Xem README mục "Phân Tích Chi Tiết".
ENTITY_LABELS = ["Lab_Value", "Medication", "Diagnosis", "Patient_Name", "Dosage"]
BASELINE_F1 = [12.1, 0.0, 0.0, 21.1, 0.0]
SINGLE_SPLIT_F1 = [100.0, 80.0, 66.7, 70.0, 30.8]  # sau khi retrain với weight_decay + EarlyStopping
OOF_F1 = [97.9, 76.9, 63.7, 62.5, 44.8]

SINGLE_SPLIT_OVERALL_F1 = 79.39  # sau khi retrain (Step 500, xem README mục Kết Quả)


def main():
    results_path = PROJECT_ROOT / "data" / "kfold_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    fold_precision = [r["precision"] * 100 for r in results["per_fold_metrics"]]
    fold_recall = [r["recall"] * 100 for r in results["per_fold_metrics"]]
    fold_f1 = [r["f1"] * 100 for r in results["per_fold_metrics"]]
    mean_f1, std_f1 = results["mean_std"]["f1"][0] * 100, results["mean_std"]["f1"][1] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ------------------------------------------------------------------
    # Biểu đồ 1: Precision / Recall / F1 qua từng fold
    # (Không có Accuracy: với NER phần lớn token là nhãn "O" nên Accuracy luôn
    #  rất cao (>90%) và không phản ánh đúng chất lượng nhận diện thực thể --
    #  Precision/Recall/F1 mới là 3 chỉ số chuẩn để đánh giá NER, xem bảng số
    #  liệu Accuracy đầy đủ ở mục "Quá trình huấn luyện chính thức" bên dưới.)
    # ------------------------------------------------------------------
    n_folds = len(fold_f1)
    x = np.arange(n_folds)
    width = 0.26
    fold_names = [f"Fold {i+1}" for i in range(n_folds)]

    ax1.bar(x - width, fold_precision, width, label="Precision", color="#DD8452", edgecolor="black", linewidth=0.6, zorder=3)
    ax1.bar(x, fold_recall, width, label="Recall", color="#55A868", edgecolor="black", linewidth=0.6, zorder=3)
    bars_f1 = ax1.bar(x + width, fold_f1, width, label="F1-Score", color="#8172B2", edgecolor="black", linewidth=0.6, zorder=3)
    for bar, val in zip(bars_f1, fold_f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ax1.axhline(mean_f1, color="#C44E52", linestyle="--", linewidth=1.5, label=f"F1 TB 5-Fold: {mean_f1:.2f}% ± {std_f1:.2f}%", zorder=2)

    ax1.set_title("Precision / Recall / F1 qua 5-Fold Cross-Validation", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Điểm số (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_names)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax1.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.95)

    # ------------------------------------------------------------------
    # Biểu đồ 2: F1 theo loại thực thể -- Baseline vs 1 lần chia vs Out-of-fold
    # ------------------------------------------------------------------
    x = np.arange(len(ENTITY_LABELS))
    width = 0.26

    ax2.bar(x - width, BASELINE_F1, width, label="Baseline rule-based (9 mẫu)", color="#CCB974", edgecolor="black", linewidth=0.6)
    ax2.bar(x, SINGLE_SPLIT_F1, width, label="LayoutLMv3, 1 lần chia (9 mẫu)", color="#8172B2", edgecolor="black", linewidth=0.6)
    ax2.bar(x + width, OOF_F1, width, label="LayoutLMv3, 5-Fold out-of-fold (45 mẫu)", color="#55A868", edgecolor="black", linewidth=0.6)

    ax2.set_title("F1 theo loại thực thể: Baseline vs. Mô hình", fontsize=13, fontweight="bold")
    ax2.set_ylabel("F1-Score (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ENTITY_LABELS, rotation=15, ha="right")
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax2.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    out_path = PROJECT_ROOT / "assets" / "kfold_results.png"
    plt.savefig(out_path, dpi=300)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ K-Fold CV tại: {out_path}")


if __name__ == "__main__":
    main()
