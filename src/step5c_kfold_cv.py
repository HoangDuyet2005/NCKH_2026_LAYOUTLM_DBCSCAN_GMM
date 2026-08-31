"""
Bước 5c: Đánh giá bằng K-Fold Cross-Validation (KHÔNG cần gán thêm nhãn).

Bước 5 gốc chỉ chia dữ liệu 1 LẦN DUY NHẤT thành 36 train / 9 test (seed=42).
Với tập test chỉ 9 mẫu, chỉ cần 1-2 thực thể dự đoán sai đã làm F1 dao động
vài điểm phần trăm -- không đủ để kết luận chắc chắn. Script này dùng
**5-Fold Cross-Validation** trên CHÍNH 45 mẫu đã gán nhãn sẵn có (không cần
thêm dữ liệu mới):

  - Chia 45 mẫu thành 5 fold; lần lượt mỗi fold làm tập test (9 mẫu), 4 fold
    còn lại (36 mẫu) làm tập train -- mỗi mẫu được dùng làm test ĐÚNG 1 LẦN.
  - Vì mỗi mẫu chỉ được dự đoán bởi 1 model CHƯA TỪNG thấy nó lúc train
    (out-of-fold prediction), việc gộp toàn bộ dự đoán của 5 fold lại cho ra
    một báo cáo đánh giá trên TOÀN BỘ 45 mẫu -- đáng tin cậy hơn nhiều so với
    chỉ đánh giá trên 9 mẫu của 1 lần chia duy nhất.
  - Đồng thời báo cáo mean ± std của F1 giữa 5 fold để thấy độ ổn định
    (variance) của pipeline huấn luyện qua các lần chia dữ liệu khác nhau.

Mỗi fold dùng `run_training()` (dùng chung với Bước 5, xem step5_layoutlmv3_finetune.py)
với weight_decay + early stopping để giảm overfitting trên tập train nhỏ.
Model của từng fold KHÔNG được lưu lại (chỉ dùng để đánh giá, không dùng để
triển khai) -- model triển khai chính thức vẫn là model đã lưu ở Bước 5.

Chạy (cần GPU để chạy trong thời gian hợp lý, mất khoảng vài chục phút):
    python src/step5c_kfold_cv.py
"""
import json
import sys
import numpy as np
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import KFold
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.metrics import f1_score as seqeval_f1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step5_layoutlmv3_finetune import load_data_from_json, run_training  # noqa: E402

N_SPLITS = 5
MAX_STEPS_PER_FOLD = 800
EVAL_STEPS = 50
EARLY_STOPPING_PATIENCE = 4


def main():
    data_list = load_data_from_json()
    if not data_list:
        print("Không có dữ liệu để đánh giá K-Fold.")
        return

    n = len(data_list)
    print(f"Tổng số mẫu gán nhãn: {n} -> chia {N_SPLITS}-Fold Cross-Validation\n")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    indices = np.arange(n)

    fold_overall_metrics = []  # [{precision, recall, f1, accuracy}, ...] theo từng fold
    oof_true, oof_pred = [], []  # gộp out-of-fold predictions của TOÀN BỘ 45 mẫu

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
        print("=" * 70)
        print(f"FOLD {fold_id}/{N_SPLITS} -- train: {len(train_idx)} mẫu | test: {len(test_idx)} mẫu")
        print("=" * 70)

        train_records = [data_list[i] for i in train_idx]
        test_records = [data_list[i] for i in test_idx]

        train_dataset_raw = Dataset.from_list(train_records)
        eval_dataset_raw = Dataset.from_list(test_records)

        run_output_dir = str(PROJECT_ROOT / "runs" / f"kfold_fold{fold_id}")

        eval_metrics, true_seqs, pred_seqs = run_training(
            train_dataset_raw,
            eval_dataset_raw,
            run_output_dir=run_output_dir,
            max_steps=MAX_STEPS_PER_FOLD,
            eval_steps=EVAL_STEPS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            save_dir=None,  # Không lưu model của từng fold, chỉ dùng để đánh giá
            disable_tqdm=True,
        )

        fold_result = {
            "fold": fold_id,
            "precision": eval_metrics.get("eval_precision"),
            "recall": eval_metrics.get("eval_recall"),
            "f1": eval_metrics.get("eval_f1"),
            "accuracy": eval_metrics.get("eval_accuracy"),
        }
        fold_overall_metrics.append(fold_result)
        oof_true.extend(true_seqs)
        oof_pred.extend(pred_seqs)

        print(f"-> Fold {fold_id} F1: {fold_result['f1']:.4f}")

    # ------------------------------------------------------------------
    # Tổng hợp kết quả
    # ------------------------------------------------------------------
    f1s = [r["f1"] for r in fold_overall_metrics]
    precisions = [r["precision"] for r in fold_overall_metrics]
    recalls = [r["recall"] for r in fold_overall_metrics]
    accs = [r["accuracy"] for r in fold_overall_metrics]

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ {N_SPLITS}-FOLD CROSS-VALIDATION (mean ± std qua các fold)")
    print("=" * 70)
    for name, vals in [("Precision", precisions), ("Recall", recalls), ("F1", f1s), ("Accuracy", accs)]:
        print(f"{name:10s}: {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}%  (từng fold: {[round(v*100,1) for v in vals]})")

    print("\n" + "=" * 70)
    print(f"BÁO CÁO OUT-OF-FOLD TRÊN TOÀN BỘ {n} MẪU (mỗi mẫu được dự đoán đúng 1 lần")
    print("bởi model CHƯA từng thấy nó lúc train) -- đáng tin cậy hơn 1 lần chia 9 mẫu")
    print("=" * 70)
    print(seqeval_report(oof_true, oof_pred, digits=3, zero_division=0))

    oof_precision = seqeval_precision(oof_true, oof_pred)
    oof_recall = seqeval_recall(oof_true, oof_pred)
    oof_f1 = seqeval_f1(oof_true, oof_pred)

    summary = {
        "n_splits": N_SPLITS,
        "max_steps_per_fold": MAX_STEPS_PER_FOLD,
        "per_fold_metrics": fold_overall_metrics,
        "mean_std": {
            "precision": [float(np.mean(precisions)), float(np.std(precisions))],
            "recall": [float(np.mean(recalls)), float(np.std(recalls))],
            "f1": [float(np.mean(f1s)), float(np.std(f1s))],
            "accuracy": [float(np.mean(accs)), float(np.std(accs))],
        },
        "out_of_fold_overall": {
            "precision": float(oof_precision),
            "recall": float(oof_recall),
            "f1": float(oof_f1),
        },
    }
    out_path = PROJECT_ROOT / "data" / "kfold_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tổng hợp tại: {out_path}")


if __name__ == "__main__":
    main()
