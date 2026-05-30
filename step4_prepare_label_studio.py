import os
import glob
import json
from PIL import Image
from tqdm import tqdm

def prepare_label_studio_format(dataset_dir: str, output_file: str = "label_studio_import.json"):
    """
    Chuyển đổi dữ liệu JSON OCR đã làm sạch thành định dạng chuẩn của Label Studio.
    BBox được chuẩn hóa theo tỷ lệ phần trăm (0-100) để hiển thị trên Label Studio.
    """
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True))
    
    label_studio_data = []
    
    for img_path in tqdm(image_paths, desc="Đang chuẩn bị dữ liệu Label Studio"):
        try:
            json_path = os.path.splitext(img_path)[0] + '.json'
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            ocr_results = data.get("ocr_results_cleaned", [])
            if not ocr_results:
                continue
            
            # Mở ảnh để lấy kích thước phục vụ tính toán phần trăm
            image = Image.open(img_path)
            width, height = image.size
            
            # Giả lập đường dẫn web (URL) cho ảnh (Label Studio cần URL hoặc local path hợp lệ)
            # Dùng Python HTTP Server ở cổng 8081 để host ảnh thay vì phụ thuộc vào tính năng bảo mật của Label Studio
            rel_img_path = img_path.replace("\\", "/")
            import time
            image_url = f"http://localhost:8081/{rel_img_path}?t={int(time.time())}"
            
            results = []
            for item in ocr_results:
                x_min, y_min, x_max, y_max = item["bbox"]
                text = item["text"]
                
                # Chuyển đổi sang tỷ lệ phần trăm cho Label Studio (0 - 100)
                x_pct = (x_min / width) * 100
                y_pct = (y_min / height) * 100
                width_pct = ((x_max - x_min) / width) * 100
                height_pct = ((y_max - y_min) / height) * 100
                
                # Cấu trúc của Label Studio BBox
                result = {
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": width_pct,
                        "height": height_pct,
                        "rotation": 0,
                        "text": [text]
                    },
                    "id": f"bbox_{x_min}_{y_min}",
                    "from_name": "transcription",
                    "to_name": "image",
                    "type": "textarea",
                    "origin": "manual"
                }
                results.append(result)
            
            task = {
                "data": {
                    "image": image_url
                },
                "predictions": [{
                    "result": results
                }]
            }
            label_studio_data.append(task)
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {img_path}: {e}")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(label_studio_data, f, ensure_ascii=False, indent=4)
    print(f"\nĐã lưu file dữ liệu Label Studio tại: {output_file}")

if __name__ == "__main__":
    DATASET_DIR = "dataset"
    prepare_label_studio_format(DATASET_DIR)
    print("Hoàn tất bước 4: Chuẩn bị dữ liệu cho Label Studio.")
