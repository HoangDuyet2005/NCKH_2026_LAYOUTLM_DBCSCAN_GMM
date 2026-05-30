import os
import glob
import json
import easyocr
from pathlib import Path
from tqdm import tqdm

def process_images_with_easyocr(dataset_dir: str):
    """
    Quét thư mục dataset và trích xuất OCR cho từng ảnh,
    lưu kết quả dưới dạng file JSON cùng cấp với ảnh.
    """
    # Khởi tạo mô hình EasyOCR cho tiếng Việt, bật chế độ sử dụng GPU nếu có
    print("Đang khởi tạo EasyOCR (ngôn ngữ: vi)...")
    reader = easyocr.Reader(['vi'], gpu=True)
    
    # Tìm tất cả các file ảnh trong thư mục dataset (bao gồm cả thư mục con)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True))
    
    print(f"Tìm thấy {len(image_paths)} ảnh cần xử lý.")
    
    for img_path in tqdm(image_paths, desc="Đang trích xuất OCR"):
        try:
            # Nhận dạng văn bản
            results = reader.readtext(img_path)
            
            ocr_data = []
            for (bbox, text, prob) in results:
                # bbox của easyocr có dạng [[x_top_left, y_top_left], [x_top_right, y_top_right], 
                #                           [x_bottom_right, y_bottom_right], [x_bottom_left, y_bottom_left]]
                x_min = int(min([pt[0] for pt in bbox]))
                y_min = int(min([pt[1] for pt in bbox]))
                x_max = int(max([pt[0] for pt in bbox]))
                y_max = int(max([pt[1] for pt in bbox]))
                
                ocr_data.append({
                    "text": text,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": float(prob)
                })
            
            # Tạo đường dẫn lưu file JSON
            json_path = os.path.splitext(img_path)[0] + '.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"ocr_results": ocr_data}, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

if __name__ == "__main__":
    DATASET_DIR = "dataset" # Thư mục gốc chứa 3 thư mục con
    process_images_with_easyocr(DATASET_DIR)
    print("Hoàn tất bước 1: Trích xuất OCR.")
