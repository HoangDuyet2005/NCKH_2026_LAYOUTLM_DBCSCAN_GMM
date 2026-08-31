import os
import glob
import json
import shutil
import torch
from PIL import Image
from datasets import Dataset
import evaluate
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.data.data_collator import default_data_collator
from pathlib import Path
import numpy as np

# Xác định thư mục gốc dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Định nghĩa hệ nhãn theo yêu cầu (BIO Format)
LABELS = [
    "O",
    "B-Patient_Name", "I-Patient_Name",
    "B-Diagnosis", "I-Diagnosis",
    "B-Medication", "I-Medication",
    "B-Dosage", "I-Dosage",
    "B-Lab_Value", "I-Lab_Value"
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

def normalize_bbox(bbox, width, height):
    """ Chuẩn hóa bounding box về tỷ lệ [0, 1000] cho LayoutLMv3 """
    x_min, y_min, x_max, y_max = bbox
    # Clip các giá trị nằm trong đoạn [0, 1000]
    return [
        max(0, min(1000, int(1000 * (x_min / width)))),
        max(0, min(1000, int(1000 * (y_min / height)))),
        max(0, min(1000, int(1000 * (x_max / width)))),
        max(0, min(1000, int(1000 * (y_max / height))))
    ]

def load_data_from_json(json_file: str = None):
    """
    Tải dữ liệu huấn luyện đã được tạo ra từ bước 4b (training_data.json).
    """
    if json_file is None:
        json_file = str(PROJECT_ROOT / "data" / "training_data.json")

    if not os.path.exists(json_file):
        print(f"[LỖI] Không tìm thấy file {json_file}. Vui lòng chạy bước 4b trước.")
        return []

    with open(json_file, 'r', encoding='utf-8') as f:
        training_records = json.load(f)

    data_list = []
    for record in training_records:
        img_path = record["image_path"]
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        data_list.append({
            "id": img_path,
            "image": image,
            "tokens": record["tokens"],
            "bboxes": record["bboxes"],
            "ner_tags": record["ner_tags"]
        })

    return data_list

# Khởi tạo Processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def encode_dataset(examples):
    """
    Hàm mã hóa dataset dùng LayoutLMv3Processor.
    LƯU Ý: Phải nạp cả ảnh gốc kết hợp Text và BBox chuẩn hóa [0, 1000].
    TUYỆT ĐỐI KHÔNG pad bboxes thủ công. Hàm của HF sẽ xử lý độ dài linh hoạt.
    """
    images = examples['image']
    words = examples['tokens']
    boxes = examples['bboxes']
    word_labels = examples['ner_tags']

    encoding = processor(
        images,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=False
    )

    return encoding


def compute_metrics(p):
    """Tính precision/recall/f1/accuracy TỔNG (overall) bằng seqeval -- dùng làm
    `metric_for_best_model` trong lúc training. Để xem chi tiết theo từng loại
    thực thể, dùng `src/step5b_eval_breakdown.py` sau khi train xong."""
    metric = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Bỏ qua nhãn đặc biệt (-100)
    true_predictions = [
        [LABELS[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABELS[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def run_training(
    train_dataset_raw,
    eval_dataset_raw,
    run_output_dir: str,
    max_steps: int = 1000,
    eval_steps: int = 100,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    save_dir: str = None,
    disable_tqdm: bool = False,
):
    """
    Hàm huấn luyện LayoutLMv3ForTokenClassification DÙNG CHUNG cho cả:
      - Bước 5 (huấn luyện chính thức, 1 lần chia 80/20) qua `finetune_layoutlmv3()`.
      - Bước 5c (K-Fold Cross-Validation) qua `step5c_kfold_cv.py`, gọi lại hàm
        này nhiều lần với các fold train/eval khác nhau, KHÔNG lặp lại logic.

    `weight_decay` và `early_stopping_patience` là 2 cải tiến chống overfitting
    (bổ sung sau khi quan sát thấy eval loss tăng nhẹ sau Step 500 trong lần
    huấn luyện gốc trên tập nhỏ 36 mẫu train -- xem mục Hạn Chế trong README).

    Trả về: (trainer đã train xong, eval_metrics dict, true_bio_seqs, pred_bio_seqs)
    true/pred_bio_seqs dùng để tính báo cáo chi tiết theo từng loại thực thể (seqeval).
    """
    print(f"Đang mã hóa dữ liệu ({len(train_dataset_raw)} train / {len(eval_dataset_raw)} eval)...")
    train_dataset = train_dataset_raw.map(encode_dataset, batched=True, remove_columns=train_dataset_raw.column_names)
    eval_dataset = eval_dataset_raw.map(encode_dataset, batched=True, remove_columns=eval_dataset_raw.column_names)
    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        id2label=id2label,
        label2id=label2id,
        num_labels=len(LABELS)
    )

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,  # Regularization L2 -- giảm overfitting trên tập train nhỏ
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,  # Giảm từ 2 -> 1 để hạn chế dung lượng checkpoint tích luỹ
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        disable_tqdm=disable_tqdm,
        report_to=[],
    )

    callbacks = []
    if early_stopping_patience:
        # Dừng sớm nếu F1 trên eval không cải thiện sau N lần eval liên tiếp,
        # tránh lãng phí compute huấn luyện tiếp khi mô hình đã bắt đầu overfit.
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()

    # Lấy predictions thô để dựng báo cáo chi tiết theo từng loại thực thể (seqeval)
    raw_predictions, raw_labels, _ = trainer.predict(eval_dataset)
    pred_ids = np.argmax(raw_predictions, axis=2)
    true_bio_seqs = [
        [LABELS[l] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(pred_ids, raw_labels)
    ]
    pred_bio_seqs = [
        [LABELS[p] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(pred_ids, raw_labels)
    ]

    if save_dir:
        trainer.model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

    # Giải phóng bộ nhớ GPU trước khi chạy fold/run tiếp theo
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dọn thư mục checkpoint trung gian (optimizer state + model mỗi checkpoint có
    # thể nặng >1GB/checkpoint). Artifact cần giữ lại đã được lưu riêng vào `save_dir`
    # ở trên (nếu có) -- checkpoint trong run_output_dir chỉ để phục vụ
    # load_best_model_at_end trong lúc train, không cần giữ sau khi train xong.
    # QUAN TRỌNG khi chạy nhiều lượt liên tiếp (vd K-Fold CV): không dọn sẽ nhanh
    # chóng làm đầy ổ đĩa (từng gây crash thật khi chạy 5-Fold CV, xem README).
    shutil.rmtree(run_output_dir, ignore_errors=True)

    return eval_metrics, true_bio_seqs, pred_bio_seqs


def finetune_layoutlmv3():
    training_data_path = str(PROJECT_ROOT / "data" / "training_data.json")
    data_list = load_data_from_json(training_data_path)
    if not data_list:
        print("Không có dữ liệu để huấn luyện.")
        return

    dataset = Dataset.from_list(data_list)

    # Cấu hình 80/20 train/test split (đây là 1 lần chia cố định seed=42 dùng làm
    # kết quả chính thức công bố trong README; xem thêm `step5c_kfold_cv.py` để có
    # đánh giá 5-fold cross-validation đáng tin cậy hơn trên cùng tập dữ liệu này).
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset_raw = dataset['train']
    eval_dataset_raw = dataset['test']

    output_dir = str(PROJECT_ROOT / "layoutlmv3-medical")
    finetuned_dir = str(PROJECT_ROOT / "layoutlmv3-medical-finetuned")

    print("Bắt đầu huấn luyện (Bước 5)...")
    eval_metrics, _, _ = run_training(
        train_dataset_raw,
        eval_dataset_raw,
        run_output_dir=output_dir,
        max_steps=1000,
        eval_steps=100,
        save_dir=finetuned_dir,
    )

    print("Huấn luyện hoàn tất và mô hình đã được lưu.")
    print("Eval metrics (best checkpoint):", eval_metrics)

if __name__ == "__main__":
    finetune_layoutlmv3()
