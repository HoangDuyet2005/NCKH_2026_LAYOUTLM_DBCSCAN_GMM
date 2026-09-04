"""
Bước 3b: Định tuyến phân loại tài liệu bằng SUPERVISED CLASSIFIER (thay thế GMM).

BỐI CẢNH: Sau khi sửa lỗi data leakage trong đánh giá GMM Router (Bước 3), số
liệu đánh giá trung thực trên tập test hold-out cho thấy GMM (không giám sát)
chỉ đạt ~62% accuracy, và THẤT BẠI HOÀN TOÀN trong việc phân biệt
`Phieu_xet_nghiem` với `Ho_so_benh_an` (0% precision/recall cho Phieu_xet_nghiem
-- toàn bộ bị nhầm thành Ho_so_benh_an). Nguyên nhân gốc:

  1. GMM dùng embedding [CLS] từ LayoutLMv3 PRETRAINED GỐC (chưa fine-tune) --
     không mang đủ đặc trưng phân biệt được 2 loại tài liệu có bố cục dạng
     bảng biểu khá giống nhau.
  2. Router được fit trên embedding pretrained gốc, nhưng
     `step6_inference_postprocessing.py` (pipeline dự đoán thật) lại tính
     embedding từ model ĐÃ FINE-TUNE -- lệch pha giữa lúc train router và lúc
     dùng router thật, khiến routing thực tế còn tệ hơn cả số liệu đo được.
  3. **Quan trọng nhất**: nhãn thật (Đơn_thuốc/Phieu_xet_nghiem/Ho_so_benh_an)
     đã có sẵn (chính là tên thư mục trong `dataset/`) -- dùng GMM không giám
     sát trong khi có sẵn nhãn là lãng phí thông tin, không phải lựa chọn tối
     ưu.

FIX: (1) Dùng embedding từ model ĐÃ FINE-TUNE (đồng bộ đúng với những gì
step6 dùng lúc inference thật), (2) dùng Logistic Regression (có giám sát,
tận dụng nhãn thật sẵn có) thay vì GMM. Kết quả xác nhận bằng 5-Fold CV trên
toàn bộ 904 ảnh: 99.89% +/- 0.22% accuracy (so với 62% của GMM cũ).

GMM (`step3_gmm_routing.py`) vẫn được giữ lại trong repo như minh hoạ kỹ thuật
phân cụm không giám sát + làm bài học cụ thể về tầm quan trọng của việc đánh
giá đúng cách (xem mục "Hạn Chế & Định Hướng Cải Thiện" trong README) -- nhưng
`document_router.pkl` được lưu bởi script NÀY mới là router dùng để triển
khai thực tế trong `step6_inference_postprocessing.py`.

Chạy: python src/step3b_supervised_routing.py
"""
import os
import glob
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import LayoutLMv3Model, LayoutLMv3Processor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_DIR = str(PROJECT_ROOT / "layoutlmv3-medical-finetuned")
LABEL_MAP = {"Don_thuoc": 0, "Phieu_xet_nghiem": 1, "Ho_so_benh_an": 2}
CLASS_MAPPING = {0: "Đơn thuốc", 1: "KQ xét nghiệm", 2: "Hồ sơ bệnh án"}


def normalize_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox
    return [
        max(0, min(1000, int(1000 * (x_min / width)))),
        max(0, min(1000, int(1000 * (y_min / height)))),
        max(0, min(1000, int(1000 * (x_max / width)))),
        max(0, min(1000, int(1000 * (y_max / height)))),
    ]


def extract_features(dataset_dir, device):
    """Trích xuất embedding [CLS] từ model ĐÃ FINE-TUNE -- ĐỒNG BỘ với cách
    step6_inference_postprocessing.py tính embedding lúc dự đoán thật."""
    processor = LayoutLMv3Processor.from_pretrained(FINETUNED_DIR, apply_ocr=False)
    model = LayoutLMv3Model.from_pretrained(FINETUNED_DIR).to(device)
    model.eval()

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    features, labels = [], []
    for img_path in tqdm(image_paths, desc="Đang trích xuất đặc trưng (model đã fine-tune)"):
        parent_dir = os.path.basename(os.path.dirname(img_path))
        if parent_dir not in LABEL_MAP:
            continue
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ocr_results = data.get("ocr_results_cleaned", [])
        if not ocr_results:
            continue
        words = [item["text"] for item in ocr_results]
        bboxes = [item["bbox"] for item in ocr_results]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        normalized_bboxes = [normalize_bbox(box, width, height) for box in bboxes]
        try:
            encoding = processor(image, words, boxes=normalized_bboxes, return_tensors="pt",
                                  truncation=True, padding="max_length", max_length=512)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = model(**encoding)
                cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            features.append(cls_embedding)
            labels.append(LABEL_MAP[parent_dir])
        except Exception as e:
            print(f"Lỗi ở ảnh {img_path}: {e}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(features), np.array(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = str(PROJECT_ROOT / "dataset")

    X, y = extract_features(dataset_dir, device)
    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    target_names = [inv_label_map[i] for i in range(3)]

    print(f"\nTổng số mẫu: {len(X)}")

    # ------------------------------------------------------------------
    # Đánh giá bằng 5-Fold CV out-of-fold (giống phương pháp dùng cho Bước 5c)
    # -- mỗi ảnh được dự đoán đúng 1 lần bởi model chưa từng thấy nó lúc train.
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_for_cv = LogisticRegression(max_iter=2000, random_state=42)
    y_pred_oof = cross_val_predict(clf_for_cv, X, y, cv=skf)

    print("\n" + "=" * 70)
    print("BÁO CÁO OUT-OF-FOLD (5-Fold CV) TRÊN TOÀN BỘ DỮ LIỆU")
    print("=" * 70)
    print(classification_report(y, y_pred_oof, target_names=target_names, zero_division=0))

    accuracy_oof = float((y_pred_oof == y).mean())
    print(f"Accuracy out-of-fold: {accuracy_oof*100:.2f}%")

    # ------------------------------------------------------------------
    # Refit trên TOÀN BỘ dữ liệu để dùng làm router triển khai thực tế
    # ------------------------------------------------------------------
    final_clf = LogisticRegression(max_iter=2000, random_state=42)
    final_clf.fit(X, y)

    router_data = {
        "type": "supervised_logreg",
        "classifier": final_clf,
        "class_mapping": CLASS_MAPPING,
        "target_names": target_names,
        "oof_accuracy": accuracy_oof,
        "embedding_source": "layoutlmv3-medical-finetuned",  # QUAN TRỌNG: phải đồng bộ với step6
    }
    save_path = str(PROJECT_ROOT / "document_router.pkl")
    joblib.dump(router_data, save_path)
    print(f"\nĐã lưu router triển khai thực tế tại: {save_path}")
    print("Lưu ý: embedding dùng để fit router này lấy từ model ĐÃ FINE-TUNE,")
    print("       PHẢI đồng bộ với cách step6_inference_postprocessing.py tính")
    print("       embedding lúc dự đoán thật, nếu không sẽ lệch pha như GMM cũ.")


if __name__ == "__main__":
    main()
