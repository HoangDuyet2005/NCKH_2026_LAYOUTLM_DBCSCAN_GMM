import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMv3Model, LayoutLMv3Processor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from tqdm import tqdm

# Xác định thư mục gốc dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def normalize_bbox(bbox, width, height):
    """ Chuẩn hóa bounding box về tỷ lệ [0, 1000] cho LayoutLMv3 """
    x_min, y_min, x_max, y_max = bbox
    return [
        int(1000 * (x_min / width)),
        int(1000 * (y_min / height)),
        int(1000 * (x_max / width)),
        int(1000 * (y_max / height))
    ]

def train_and_evaluate_gmm(dataset_dir: str):
    print("Đang khởi tạo LayoutLMv3 Feature Extractor...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
    
    # Thiết lập device (ưu tiên GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True))
    
    features = []
    labels = []
    label_map = {"Don_thuoc": 0, "Phieu_xet_nghiem": 1, "Ho_so_benh_an": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    
    for img_path in tqdm(image_paths, desc="Đang trích xuất đặc trưng"):
        try:
            # Tìm label thực tế dựa trên tên thư mục chứa ảnh
            parent_dir = os.path.basename(os.path.dirname(img_path))
            if parent_dir not in label_map:
                continue
            
            json_path = os.path.splitext(img_path)[0] + '.json'
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            ocr_results = data.get("ocr_results_cleaned", [])
            if not ocr_results:
                continue
                
            words = [item["text"] for item in ocr_results]
            bboxes = [item["bbox"] for item in ocr_results]
            
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            normalized_bboxes = [normalize_bbox(box, width, height) for box in bboxes]
            
            # Chuẩn bị dữ liệu đầu vào cho LayoutLMv3
            encoding = processor(image, words, boxes=normalized_bboxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = model(**encoding)
                # Lấy vector [CLS] đại diện cho toàn bộ trang (Page-level Embedding)
                cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                
            features.append(cls_embedding)
            labels.append(label_map[parent_dir])
            
        except Exception as e:
            print(f"Lỗi ở ảnh {img_path}: {e}")
            
    if not features:
        print("Không có đủ dữ liệu đặc trưng để huấn luyện GMM.")
        return
        
    X = np.array(features)
    y_true = np.array(labels)

    # ---------------------------------------------------------
    # Tách Train/Test (80/20, stratified) để đánh giá KHÔNG bị lạc quan giả tạo.
    # GMM là mô hình không giám sát nên việc fit rồi đánh giá lại trên chính
    # tập đã fit sẽ luôn cho điểm số cao hơn thực tế (data leakage). Ở đây ta:
    #   1) Fit GMM + xác định ánh xạ cluster->nhãn CHỈ trên tập train.
    #   2) Đánh giá classification_report/confusion_matrix TRÊN TẬP TEST (chưa
    #      từng thấy khi fit), để có con số phản ánh đúng khả năng tổng quát hoá.
    #   3) Sau khi đã có số liệu đánh giá trung thực, refit lại GMM cuối cùng
    #      trên TOÀN BỘ dữ liệu (train+test) để tận dụng hết dữ liệu sẵn có cho
    #      bản triển khai thực tế — đây là lựa chọn thường gặp trong thực tế,
    #      miễn là số liệu báo cáo ở bước 2 không bị lẫn với model triển khai.
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=42, stratify=y_true
    )

    print(f"\nSố mẫu train: {len(X_train)} | Số mẫu test (hold-out): {len(X_test)}")
    print("Đang huấn luyện GaussianMixture Model (K=3) trên tập TRAIN...")
    gmm_eval = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    y_pred_cluster_train = gmm_eval.fit_predict(X_train)

    # Do GMM là học không giám sát, cần map lại id cluster thành nhãn chuẩn dựa trên
    # nhãn phổ biến nhất trong cluster -- ánh xạ này CHỈ được suy ra từ tập train.
    cluster_to_label = {}
    for cluster_id in range(3):
        mask = (y_pred_cluster_train == cluster_id)
        if np.any(mask):
            true_labels_in_cluster = y_train[mask]
            most_frequent_label = np.bincount(true_labels_in_cluster).argmax()
            cluster_to_label[cluster_id] = most_frequent_label
        else:
            cluster_to_label[cluster_id] = 0  # Default fallback

    target_names = [inv_label_map[i] for i in range(3)]

    # Đánh giá trên tập TEST (hold-out, GMM chưa từng thấy các điểm này)
    y_pred_cluster_test = gmm_eval.predict(X_test)
    y_pred_mapped_test = np.array([cluster_to_label[c] for c in y_pred_cluster_test])

    print("\n--- Classification Report (đánh giá trên tập TEST hold-out, KHÔNG dùng để fit) ---")
    print(classification_report(y_test, y_pred_mapped_test, target_names=target_names))

    print("\n--- Confusion Matrix (tập TEST hold-out) ---")
    print(confusion_matrix(y_test, y_pred_mapped_test))

    # ---------------------------------------------------------
    # Sau khi đã có số liệu đánh giá trung thực ở trên, refit GMM cuối cùng trên
    # TOÀN BỘ dữ liệu để dùng làm router triển khai (nhiều dữ liệu hơn -> ước lượng
    # các cụm Gaussian ổn định hơn). Ánh xạ cluster->nhãn cũng được suy lại trên
    # toàn bộ dữ liệu cho model triển khai này.
    # ---------------------------------------------------------
    print("\nĐang refit GaussianMixture Model (K=3) trên TOÀN BỘ dữ liệu để triển khai...")
    gmm_final = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    y_pred_cluster_full = gmm_final.fit_predict(X)

    cluster_to_label_full = {}
    for cluster_id in range(3):
        mask = (y_pred_cluster_full == cluster_id)
        if np.any(mask):
            true_labels_in_cluster = y_true[mask]
            most_frequent_label = np.bincount(true_labels_in_cluster).argmax()
            cluster_to_label_full[cluster_id] = most_frequent_label
        else:
            cluster_to_label_full[cluster_id] = 0

    # Lưu mô hình GMM đã huấn luyện cùng bảng ánh xạ nhãn để Step 6 nạp và định tuyến chuẩn xác
    gmm_save_path = str(PROJECT_ROOT / "gmm_router.pkl")
    router_data = {
        "gmm": gmm_final,
        "cluster_to_label": cluster_to_label_full,
        "target_names": target_names,
        "class_mapping": {0: "Đơn thuốc", 1: "KQ xét nghiệm", 2: "Hồ sơ bệnh án"}
    }
    joblib.dump(router_data, gmm_save_path)
    print(f"\nĐã lưu mô hình GMM định tuyến (huấn luyện trên toàn bộ dữ liệu) tại: {gmm_save_path}")
    print("Lưu ý: Classification Report/Confusion Matrix ở trên được đo trên tập TEST hold-out")
    print("       (không dùng để fit), phản ánh đúng khả năng tổng quát hoá của router.")

if __name__ == "__main__":
    DATASET_DIR = str(PROJECT_ROOT / "dataset")
    train_and_evaluate_gmm(DATASET_DIR)
    print("Hoàn tất bước 3: Định tuyến phân loại tài liệu với GMM.")
