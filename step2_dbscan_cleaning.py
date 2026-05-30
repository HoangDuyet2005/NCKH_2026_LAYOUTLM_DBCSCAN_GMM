import os
import glob
import json
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from tqdm import tqdm

def clean_ocr_with_dbscan(json_path: str, eps: float = 15.0, min_samples: int = 1):
    """
    Đọc file JSON chứa kết quả OCR, tính toán tọa độ trọng tâm (center_x, center_y),
    dùng DBSCAN để gom cụm các từ trên cùng dòng và loại bỏ điểm nhiễu.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        ocr_results = data.get("ocr_results", [])
        if not ocr_results:
            return
            
        centers = []
        for item in ocr_results:
            x_min, y_min, x_max, y_max = item["bbox"]
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            centers.append([center_x, center_y])
            
        X = np.array(centers)
        
        # Ở đây ta muốn gom cụm theo dòng (ưu tiên khoảng cách y nhỏ, x có thể xa hơn)
        # Ta có thể scale lại tọa độ x để giảm ảnh hưởng của khoảng cách x trong DBSCAN
        X_scaled = X.copy()
        X_scaled[:, 0] = X_scaled[:, 0] * 0.1 # Thu nhỏ x đi 10 lần
        
        # Áp dụng DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        cleaned_results = []
        for i, label in enumerate(labels):
            # label == -1 là điểm nhiễu (outlier) trong thuật toán DBSCAN
            if label != -1:
                item = ocr_results[i].copy()
                item["cluster_id"] = int(label)
                cleaned_results.append(item)
                
        # Sắp xếp các phần tử trong cùng một cụm (dòng) theo tọa độ x tăng dần
        cleaned_results.sort(key=lambda item: (item["cluster_id"], item["bbox"][0]))
        
        # Cập nhật file JSON
        data["ocr_results_cleaned"] = cleaned_results
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_path}: {e}")

def process_all_jsons(dataset_dir: str):
    json_paths = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
    print(f"Tìm thấy {len(json_paths)} file JSON cần làm sạch.")
    
    for json_path in tqdm(json_paths, desc="Đang chạy DBSCAN cleaning"):
        clean_ocr_with_dbscan(json_path)

if __name__ == "__main__":
    DATASET_DIR = "dataset"
    process_all_jsons(DATASET_DIR)
    print("Hoàn tất bước 2: Làm sạch và gom dòng với DBSCAN.")
