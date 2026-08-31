import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Xác định thư mục gốc dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    # Đọc từ `trainer_log_history.json` được lưu cùng model đã fine-tune
    # (layoutlmv3-medical-finetuned/), KHÔNG đọc từ thư mục checkpoint trung gian
    # (layoutlmv3-medical/checkpoint-*) -- thư mục đó bị `run_training()` tự dọn
    # sau khi train xong (tránh làm đầy ổ đĩa, xem step5_layoutlmv3_finetune.py),
    # nên log_history phải được lưu lại ở nơi bền vững trước khi checkpoint mất đi.
    log_path = str(PROJECT_ROOT / "layoutlmv3-medical-finetuned" / "trainer_log_history.json")
    if not os.path.exists(log_path):
        print(f"Error: Could not find {log_path}. Hãy chạy lại src/step5_layoutlmv3_finetune.py trước.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log_history = json.load(f)

    # Extract eval metrics and train metrics
    eval_history = []
    train_history = []

    for entry in log_history:
        if "eval_loss" in entry:
            eval_history.append({
                "step": entry["step"],
                "loss": entry["eval_loss"],
                "precision": entry.get("eval_precision", 0),
                "recall": entry.get("eval_recall", 0),
                "f1": entry.get("eval_f1", 0),
                "accuracy": entry.get("eval_accuracy", 0),
                "epoch": entry.get("epoch", 0)
            })
        elif "loss" in entry:
            train_history.append({
                "step": entry["step"],
                "loss": entry["loss"]
            })

    # 1. Print Markdown Table
    print("\n" + "="*40)
    print(" BẢNG THÔNG SỐ ĐÁNH GIÁ QUA CÁC BƯỚC (EVALUATION METRICS)")
    print("="*40)
    print("| Step | Epoch | Eval Loss | Precision (%) | Recall (%) | F1-Score (%) | Accuracy (%) |")
    print("|------|-------|-----------|---------------|------------|--------------|--------------|")
    for row in eval_history:
        print(f"| {row['step']:4d} | {row['epoch']:5.2f} | {row['loss']:.4f}    | {row['precision']*100:12.2f} | {row['recall']*100:9.2f} | {row['f1']*100:12.2f} | {row['accuracy']*100:11.2f} |")
    
    best_row = max(eval_history, key=lambda row: row["f1"]) if eval_history else None

    print("\n" + "="*40)
    if best_row:
        print(f"Chỉ số tốt nhất (Best F1-Score): {best_row['f1']*100:.2f}% tại Step {best_row['step']}")
    print("="*40)

    # 2. Plotting the metrics
    steps = [row["step"] for row in eval_history]
    eval_loss = [row["loss"] for row in eval_history]
    precision = [row["precision"] for row in eval_history]
    recall = [row["recall"] for row in eval_history]
    f1 = [row["f1"] for row in eval_history]
    accuracy = [row["accuracy"] for row in eval_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left Plot: Loss Curves
    ax1.plot(steps, eval_loss, label="Validation Loss", color="red", marker="o", linewidth=2)
    if train_history:
        train_steps = [row["step"] for row in train_history]
        train_loss = [row["loss"] for row in train_history]
        ax1.plot(train_steps, train_loss, label="Training Loss", color="blue", linestyle="--", marker="s", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Right Plot: Performance Metrics
    ax2.plot(steps, [p*100 for p in precision], label="Precision", color="orange", marker="^", linewidth=2)
    ax2.plot(steps, [r*100 for r in recall], label="Recall", color="green", marker="v", linewidth=2)
    ax2.plot(steps, [f*100 for f in f1], label="F1-Score", color="purple", marker="D", linewidth=2)
    ax2.plot(steps, [a*100 for a in accuracy], label="Accuracy", color="brown", marker="x", linestyle="-.", linewidth=1.5)
    
    # Highlight best model checkpoint (xác định động từ eval_history, không hardcode)
    if best_row:
        best_step, best_f1 = best_row["step"], best_row["f1"] * 100
        ax2.axvline(x=best_step, color="red", linestyle=":", label=f"Best Model (Step {best_step})")
        ax2.annotate(f"Best F1: {best_f1:.1f}%", xy=(best_step, best_f1), xytext=(best_step+50, best_f1-5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    ax2.set_title("Performance Metrics (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.set_ylim(30, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    chart_output = str(PROJECT_ROOT / "assets" / "training_progress.png")
    plt.savefig(chart_output, dpi=300)
    print(f"\n[THÀNH CÔNG] Đã lưu biểu đồ kết quả huấn luyện tại: {os.path.abspath(chart_output)}")

if __name__ == "__main__":
    main()
