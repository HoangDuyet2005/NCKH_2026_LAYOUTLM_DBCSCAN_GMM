import os
import glob
import json
import torch
from PIL import Image
from datasets import Dataset
import evaluate
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification, 
    TrainingArguments, 
    Trainer
)
from transformers.data.data_collator import default_data_collator
import numpy as np

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

def load_data_from_json(json_file: str = "training_data.json"):
    """
    Tải dữ liệu huấn luyện đã được tạo ra từ bước 4b (training_data.json).
    """
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

def finetune_layoutlmv3():
    data_list = load_data_from_json("training_data.json")
    if not data_list:
        print("Không có dữ liệu để huấn luyện.")
        return
        
    dataset = Dataset.from_list(data_list)
    
    # Cấu hình 80/20 train/test split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    print("Đang mã hóa dữ liệu huấn luyện...")
    train_dataset = train_dataset.map(encode_dataset, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(encode_dataset, batched=True, remove_columns=eval_dataset.column_names)

    # Đặt định dạng PyTorch
    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    # Tính metric với seqeval
    metric = evaluate.load("seqeval")
    def compute_metrics(p):
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

    print("Đang khởi tạo mô hình LayoutLMv3ForTokenClassification...")
    # TUYỆT ĐỐI KHÔNG tự viết lớp Classifier thủ công, sử dụng cấu trúc có sẵn
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        id2label=id2label,
        label2id=label2id,
        num_labels=len(LABELS)
    )

    training_args = TrainingArguments(
        output_dir="./layoutlmv3-medical",
        max_steps=1000,
        per_device_train_batch_size=4, # Batch size = 4
        per_device_eval_batch_size=4,
        learning_rate=1e-5, # Learning rate = 1e-5
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu huấn luyện...")
    trainer.train()
    
    # Lưu mô hình
    model.save_pretrained("./layoutlmv3-medical-finetuned")
    processor.save_pretrained("./layoutlmv3-medical-finetuned")
    print("Huấn luyện hoàn tất và mô hình đã được lưu.")

if __name__ == "__main__":
    finetune_layoutlmv3()
