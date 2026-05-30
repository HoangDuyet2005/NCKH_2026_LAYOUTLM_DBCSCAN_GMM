import os
import json
import matplotlib.pyplot as plt

def main():
    state_path = os.path.join("layoutlmv3-medical", "checkpoint-1000", "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"Error: Could not find {state_path}")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    # Extract eval metrics and train metrics
    eval_history = []
    train_history = []
    
    for entry in state_data.get("log_history", []):
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
    
    print("\n" + "="*40)
    print(f"Chỉ số tốt nhất (Best F1-Score): {state_data.get('best_metric', 0)*100:.2f}% tại checkpoint: {state_data.get('best_model_checkpoint', 'N/A')}")
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
    
    # Highlight best model checkpoint (F1 = 72% at step 500)
    best_step = 500
    best_f1 = 0.72 * 100
    ax2.axvline(x=best_step, color="red", linestyle=":", label="Best Model (Step 500)")
    ax2.annotate(f"Best F1: {best_f1:.1f}%", xy=(best_step, best_f1), xytext=(best_step+50, best_f1-5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    ax2.set_title("Performance Metrics (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.set_ylim(30, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    chart_output = "training_progress.png"
    plt.savefig(chart_output, dpi=300)
    print(f"\n[THÀNH CÔNG] Đã lưu biểu đồ kết quả huấn luyện tại: {os.path.abspath(chart_output)}")

if __name__ == "__main__":
    main()
