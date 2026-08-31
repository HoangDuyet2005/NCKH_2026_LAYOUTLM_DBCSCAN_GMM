"""
Bước 5b: Phân tích sâu kết quả đánh giá trên tập dữ liệu hiện có (KHÔNG cần gán thêm nhãn).

Vì tập dữ liệu gán nhãn chỉ có 45 mẫu (36 train / 9 test), con số F1 tổng
(overall F1) duy nhất trong README dễ gây hiểu nhầm — nó gộp cả những loại
thực thể mô hình làm tốt lẫn những loại làm kém. Script này tận dụng LẠI
đúng tập dữ liệu và đúng eval split (seed=42) đã dùng ở Bước 5 để:

  1. Đánh giá CHI TIẾT THEO TỪNG LOẠI THỰC THỂ (Patient_Name, Diagnosis,
     Medication, Dosage, Lab_Value) bằng mô hình LayoutLMv3 đã fine-tune,
     thay vì chỉ nhìn 1 con số F1 tổng.
  2. Xây dựng một baseline rule-based đơn giản (regex/keyword, không học máy)
     để có đối chứng định lượng: mô hình LayoutLMv3 thực sự tốt hơn một cách
     tiếp cận "ngây thơ" bao nhiêu, ở loại thực thể nào.

Chạy:
    python src/step5b_eval_breakdown.py
"""
import re
import sys
import torch
from pathlib import Path
from datasets import Dataset
from seqeval.metrics import classification_report as seqeval_report
from transformers import LayoutLMv3ForTokenClassification
from transformers.data.data_collator import default_data_collator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tái sử dụng nguyên vẹn hàm nạp dữ liệu + processor + LABELS từ Bước 5 để
# đảm bảo encode CHÍNH XÁC giống lúc train (cùng seed => cùng eval split).
from step5_layoutlmv3_finetune import (  # noqa: E402
    LABELS, load_data_from_json, encode_dataset, processor
)

FINETUNED_DIR = str(PROJECT_ROOT / "layoutlmv3-medical-finetuned")


def build_eval_split():
    data_list = load_data_from_json()
    dataset = Dataset.from_list(data_list)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)  # PHẢI khớp Bước 5
    return dataset["train"], dataset["test"]


def run_model_predictions(eval_dataset):
    """Chạy inference bằng model đã fine-tune, trả về true/pred labels dạng BIO (list-of-list)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3ForTokenClassification.from_pretrained(FINETUNED_DIR).to(device)
    model.eval()

    encoded = eval_dataset.map(encode_dataset, batched=True, remove_columns=eval_dataset.column_names)
    encoded.set_format(type="torch")

    all_true, all_pred = [], []
    with torch.no_grad():
        for i in range(len(encoded)):
            batch = default_data_collator([encoded[i]])
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")[0].tolist()
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)[0].tolist()

            true_seq = [LABELS[l] for l, p in zip(labels, preds) if l != -100]
            pred_seq = [LABELS[p] for l, p in zip(labels, preds) if l != -100]
            all_true.append(true_seq)
            all_pred.append(pred_seq)

    return all_true, all_pred


# ---------------------------------------------------------------------------
# Baseline rule-based đơn giản (không học máy) để làm đối chứng.
# CHỈ dùng regex/keyword tổng quát -- không "học" từ tập test để tránh gian lận.
# ---------------------------------------------------------------------------
LAB_UNIT_PATTERN = re.compile(
    r"^\d+[.,]?\d*\s*(U/L|UIL|U/l|g/L|gIL|mmol/L|mg/dL|mg/dl|IU/L)?$", re.IGNORECASE
)
LAB_UNIT_TOKENS = {"U/L", "UIL", "g/L", "mmol/L", "mg/dL", "IU/L"}
NAME_TRIGGER_WORDS = {"tên", "họ", "bệnh", "nhân"}


def rule_based_predict(tokens):
    """Gán nhãn BIO bằng heuristic thuần regex/keyword cho 1 danh sách token của 1 tài liệu."""
    tags = ["O"] * len(tokens)
    for i, tok in enumerate(tokens):
        tok_clean = tok.strip()
        # Heuristic Lab_Value: token là số + đơn vị xét nghiệm phổ biến đi kèm
        has_digit = bool(re.search(r"\d", tok_clean))
        has_unit = any(u.lower() in tok_clean.lower() for u in LAB_UNIT_TOKENS)
        if has_digit and has_unit:
            tags[i] = "B-Lab_Value"
        # Heuristic Patient_Name: token ngay sau các từ khóa "tên"/"họ tên"/"bệnh nhân"
        elif i > 0 and any(w in tokens[i - 1].lower() for w in NAME_TRIGGER_WORDS) and tok_clean[:1].isupper():
            tags[i] = "B-Patient_Name" if (i < 2 or tags[i - 1] == "O") else "I-Patient_Name"
        # Diagnosis / Medication / Dosage: không có heuristic đáng tin cậy => để "O"
        # (đây chính là điểm mà một baseline "ngây thơ" thua mô hình học máy).
    return tags


def run_baseline_predictions(eval_dataset):
    all_true, all_pred = [], []
    for record in eval_dataset:
        true_seq = [LABELS[t] for t in record["ner_tags"]]
        pred_seq = rule_based_predict(record["tokens"])
        all_true.append(true_seq)
        all_pred.append(pred_seq)
    return all_true, all_pred


def main():
    train_split, eval_split = build_eval_split()
    print(f"Tập train: {len(train_split)} mẫu | Tập test (hold-out, giống Bước 5): {len(eval_split)} mẫu\n")

    print("=" * 70)
    print("1) BASELINE RULE-BASED (regex/keyword, không học máy) — đối chứng")
    print("=" * 70)
    b_true, b_pred = run_baseline_predictions(eval_split)
    print(seqeval_report(b_true, b_pred, digits=3, zero_division=0))

    print("=" * 70)
    print("2) MÔ HÌNH LAYOUTLMV3 ĐÃ FINE-TUNE — chi tiết theo từng loại thực thể")
    print("=" * 70)
    m_true, m_pred = run_model_predictions(eval_split)
    print(seqeval_report(m_true, m_pred, digits=3, zero_division=0))

    print("Lưu ý: cả 2 phần trên dùng CHUNG một eval split hold-out (9 mẫu, seed=42,")
    print("       giống hệt Bước 5) nên có thể so sánh trực tiếp baseline vs. mô hình.")
    print("       Do cỡ mẫu nhỏ, số liệu per-entity ở trên nên đọc cùng cột 'support'")
    print("       (số thực thể thật có trong tập test) để biết mức độ tin cậy.")


if __name__ == "__main__":
    main()
