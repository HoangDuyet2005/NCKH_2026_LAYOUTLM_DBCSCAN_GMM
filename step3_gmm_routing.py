import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Model, LayoutLMv3Processor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

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
    
    print("\nĐang huấn luyện GaussianMixture Model (K=3)...")
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    y_pred_cluster = gmm.fit_predict(X)
    
    # Do GMM là học không giám sát, cần map lại id cluster thành nhãn chuẩn dựa trên nhãn phổ biến nhất trong cluster
    cluster_to_label = {}
    for cluster_id in range(3):
        mask = (y_pred_cluster == cluster_id)
        if np.any(mask):
            true_labels_in_cluster = y_true[mask]
            most_frequent_label = np.bincount(true_labels_in_cluster).argmax()
            cluster_to_label[cluster_id] = most_frequent_label
        else:
            cluster_to_label[cluster_id] = 0 # Default fallback
            
    y_pred_mapped = np.array([cluster_to_label[c] for c in y_pred_cluster])
    
    # In báo cáo kết quả
    target_names = [inv_label_map[i] for i in range(3)]
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred_mapped, target_names=target_names))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred_mapped))

if __name__ == "__main__":
    DATASET_DIR = "dataset"
    train_and_evaluate_gmm(DATASET_DIR)
    print("Hoàn tất bước 3: Định tuyến phân loại tài liệu với GMM.")
