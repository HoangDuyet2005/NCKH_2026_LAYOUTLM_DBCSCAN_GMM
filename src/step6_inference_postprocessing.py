import os
import json
import re
import torch
import joblib
import numpy as np
from PIL import Image
from pathlib import Path
import easyocr
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Xác định thư mục gốc dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def normalize_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox
    return [
        max(0, min(1000, int(1000 * (x_min / width)))),
        max(0, min(1000, int(1000 * (y_min / height)))),
        max(0, min(1000, int(1000 * (x_max / width)))),
        max(0, min(1000, int(1000 * (y_max / height))))
    ]

def clean_text(text: str) -> str:
    """ Dùng Regex làm sạch các ký tự rác ở đầu/cuối chuỗi """
    return re.sub(r'^[\s|\-_,]+|[\s|\-_,]+$', '', text)

def inference_pipeline(img_path: str, model_path: str = None, router_path: str = None, gmm_path: str = None):
    if model_path is None:
        model_path = str(PROJECT_ROOT / "layoutlmv3-medical-finetuned")
    # `router_path` (document_router.pkl, src/step3b_supervised_routing.py) là router
    # chính thức dùng để triển khai -- Logistic Regression có giám sát trên embedding
    # ĐÃ FINE-TUNE, xác nhận 99.89% +/- 0.22% accuracy qua 5-Fold CV (so với ~62% của
    # GMM cũ, xem README mục Hạn Chế). `gmm_path` chỉ giữ lại làm phương án dự phòng
    # nếu router mới chưa tồn tại.
    if router_path is None:
        router_path = str(PROJECT_ROOT / "document_router.pkl")
    if gmm_path is None:
        gmm_path = str(PROJECT_ROOT / "gmm_router.pkl")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # 1. OCR -> DBSCAN Cleaning
    # ---------------------------------------------------------
    print("1. Chạy OCR và DBSCAN Cleaning...")
    reader = easyocr.Reader(['vi'], gpu=True)
    ocr_results = reader.readtext(img_path)
    
    cleaned_ocr = []
    centers = []
    for (bbox, text, prob) in ocr_results:
        x_min = int(min([pt[0] for pt in bbox]))
        y_min = int(min([pt[1] for pt in bbox]))
        x_max = int(max([pt[0] for pt in bbox]))
        y_max = int(max([pt[1] for pt in bbox]))
        
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        cleaned_ocr.append({
            "text": text,
            "bbox": [x_min, y_min, x_max, y_max]
        })
        centers.append([center_x * 0.1, center_y])
        
    if not centers:
        return {"error": "Không tìm thấy text trong ảnh."}
        
    # Áp dụng DBSCAN
    dbscan = DBSCAN(eps=15.0, min_samples=1)
    labels = dbscan.fit_predict(np.array(centers))
    
    final_ocr = []
    for i, label in enumerate(labels):
        if label != -1:
            item = cleaned_ocr[i].copy()
            item["cluster_id"] = int(label)
            final_ocr.append(item)
    
    final_ocr.sort(key=lambda item: (item["cluster_id"], item["bbox"][0]))
    
    # ---------------------------------------------------------
    # 2. Khởi tạo mô hình LayoutLMv3 và Trích xuất Page-level Embedding [CLS]
    # ---------------------------------------------------------
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    
    try:
        processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}. Vui lòng chạy huấn luyện ở Bước 5 trước.")
        return {"error": "Mô hình chưa được finetune."}
        
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 3. Document Routing (LayoutLMv3 [CLS] 768-D từ model ĐÃ FINE-TUNE -> Logistic Regression)
    # ---------------------------------------------------------
    print("2. Chạy Document Routing (định tuyến phân loại tài liệu)...")
    document_type = "Đơn thuốc" # Mặc định dự phòng

    try:
        # Chuẩn bị toàn bộ text và bboxes để trích xuất đặc trưng trang
        all_words = [item["text"] for item in final_ocr]
        all_bboxes = [normalize_bbox(item["bbox"], width, height) for item in final_ocr]

        page_enc = processor(image, all_words, boxes=all_bboxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        page_enc = {k: v.to(device) for k, v in page_enc.items()}

        with torch.no_grad():
            base_output = model.layoutlmv3(**page_enc)
            # Lấy vector [CLS] 768 chiều đại diện cho toàn bộ trang -- LƯU Ý: tính từ
            # model ĐÃ FINE-TUNE, phải đồng bộ với nguồn embedding lúc fit router
            # (xem step3b_supervised_routing.py) nếu không sẽ lệch pha.
            cls_embedding = base_output.last_hidden_state[0, 0, :].cpu().numpy().reshape(1, -1)

        if os.path.exists(router_path):
            # Router chính thức: Logistic Regression có giám sát (99.89% +/- 0.22% qua 5-Fold CV)
            router_data = joblib.load(router_path)
            clf = router_data["classifier"]
            class_mapping = router_data.get("class_mapping", {0: "Đơn thuốc", 1: "KQ xét nghiệm", 2: "Hồ sơ bệnh án"})
            pred_label_id = int(clf.predict(cls_embedding)[0])
            document_type = class_mapping.get(pred_label_id, "Đơn thuốc")
        elif os.path.exists(gmm_path):
            # Phương án dự phòng nếu chưa chạy step3b_supervised_routing.py -- LƯU Ý:
            # GMM đo được chỉ ~62% accuracy (xem README, mục Hạn Chế), không khuyến
            # khích dùng cho triển khai thực tế, chỉ giữ để tương thích ngược.
            print("-> Chưa có document_router.pkl, dùng GMM cũ (kém tin cậy hơn nhiều, ~62% accuracy).")
            router_data = joblib.load(gmm_path)
            if isinstance(router_data, dict):
                gmm_model = router_data["gmm"]
                c2l = router_data.get("cluster_to_label", {})
                class_mapping = router_data.get("class_mapping", {0: "Đơn thuốc", 1: "KQ xét nghiệm", 2: "Hồ sơ bệnh án"})
                pred_cluster = int(gmm_model.predict(cls_embedding)[0])
                pred_label_id = c2l.get(pred_cluster, pred_cluster)
                document_type = class_mapping.get(pred_label_id, "Đơn thuốc")
            else:
                pred_cluster = int(router_data.predict(cls_embedding)[0])
                document_type = {0: "Đơn thuốc", 1: "KQ xét nghiệm", 2: "Hồ sơ bệnh án"}.get(pred_cluster, "Đơn thuốc")
        else:
            print("-> Chưa có router nào (document_router.pkl / gmm_router.pkl), dùng phân loại mặc định.")
    except Exception as e:
        print(f"Lỗi khi chạy Document Router: {e}")

    print(f"-> Phân loại tài liệu: {document_type}")
    
    # ---------------------------------------------------------
    # 4. LayoutLMv3 Token Classification (Trích xuất thực thể theo Chunks)
    # ---------------------------------------------------------
    print("3. Trích xuất thực thể với LayoutLMv3...")
    id2label = model.config.id2label
    raw_entities = []
    current_entity = None
    
    # THUẬT TOÁN PHÂN CHUNK THEO ĐỘ DÀI TOKEN THỰC TẾ (BẢO TOÀN ĐỀ TÀI KHÔNG BỊ CHẶT MẤT CHỮ)
    chunks = []
    current_chunk = []
    max_allowed_tokens = 420  # Ngưỡng tối ưu loại trừ token ảnh rác
    
    for item in final_ocr:
        test_words = [chk["text"] for chk in current_chunk] + [item["text"]]
        test_bboxes = [normalize_bbox(chk["bbox"], width, height) for chk in current_chunk] + [normalize_bbox(item["bbox"], width, height)]
        
        test_enc = processor(image, test_words, boxes=test_bboxes, return_tensors="pt")
        total_tokens = test_enc.input_ids.shape[1]
        
        if total_tokens > max_allowed_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [item]
        else:
            current_chunk.append(item)
            
    if current_chunk:
        chunks.append(current_chunk)
        
    print(f"-> Phân tách cấu trúc tài liệu thành {len(chunks)} Chunks để nạp mượt vào bộ nhớ LayoutLMv3.")
    
    for chunk_items in chunks:
        chunk_words = [item["text"] for item in chunk_items]
        chunk_bboxes = [normalize_bbox(item["bbox"], width, height) for item in chunk_items]
        
        # Giữ nguyên đối tượng encoding gốc để trích xuất word_ids() không bị lỗi biến dictionary độc lập
        encoding = processor(image, chunk_words, boxes=chunk_bboxes, return_tensors="pt")
        word_ids = encoding.word_ids()
        
        # Chỉ chuyển đổi lớp dữ liệu tính toán toán học lên thiết bị GPU chuyên dụng
        model_inputs = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
            if isinstance(predictions, int):
                predictions = [predictions]
                
        # ---------------------------------------------------------
        # MỤC 2.4.1: Hậu xử lý (Post-processing) và gộp nhãn BIO giữa các Chunks liên tục
        # ---------------------------------------------------------
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
                
            label = id2label[predictions[idx]]
            word = chunk_words[word_idx]
            
            if label.startswith("B-"):
                if current_entity:
                    current_entity["text"] = clean_text(current_entity["text"]) # Làm sạch Regex
                    raw_entities.append(current_entity)
                current_entity = {
                    "type": label[2:],
                    "text": word
                }
            elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
                current_entity["text"] += " " + word # Gộp nhãn I-
            else:
                if current_entity:
                    current_entity["text"] = clean_text(current_entity["text"]) # Làm sạch Regex
                    raw_entities.append(current_entity)
                    current_entity = None
                    
            previous_word_idx = word_idx
            
    if current_entity:
        current_entity["text"] = clean_text(current_entity["text"])
        raw_entities.append(current_entity)

    # ---------------------------------------------------------
    # 4. HẬU XỬ LÝ VÀ CẤU TRÚC HÓA DỮ LIỆU ĐẦU RA (Tương ứng Mục 2.4)
    # ---------------------------------------------------------
    print("4. Chạy Module Hậu xử lý (Mã hóa, Z-Score và Xử lý Khuyết thiếu)...")
    
    field_mapping = {
        "Patient_Name": "Tên bệnh nhân",
        "Diagnosis": "Chẩn đoán",
        "Medication": "Tên thuốc",
        "Dosage": "Liều lượng",
        "Lab_Value": "Kết quả xét nghiệm"
    }
    
    found_types = set([ent["type"] for ent in raw_entities])
    formatted_entities = []
    
    for ent in raw_entities:
        e_type = ent["type"]
        e_text = ent["text"]
        field_name = field_mapping.get(e_type, e_type)
        
        # MỤC 2.4.3: SỐ HÓA VÀ ĐỒNG NHẤT QUY MÔ ĐẶC TRƯNG
        entity_data = {"field": field_name, "value": e_text}
        
        if e_type == "Lab_Value":
            # Đồng nhất thang đo bằng chuẩn hóa Z-Score
            baseline_stats = {"U/L": (40.0, 15.0), "g/L": (30.0, 5.0), "mmol/L": (5.5, 1.2)}
            match = re.match(r"([\d\.,]+)\s*(.*)", e_text.strip())
            if match:
                num_str = match.group(1).replace(',', '.')
                unit = match.group(2).strip().replace("UIL", "U/L")
                try:
                    X_val = float(num_str)
                    if unit in baseline_stats:
                        mu, sigma = baseline_stats[unit]
                        z_score = round((X_val - mu) / sigma, 3) # Áp dụng công thức X_norm = (X - mu) / sigma
                        entity_data["Z_Score_Normalized"] = z_score
                except ValueError:
                    pass
        elif e_type in ["Diagnosis", "Medication"]:
            # Mã hóa đặc trưng phân loại (One-Hot Encoding)
            vocab_space = ["Tiểu đường", "Huyết áp cao", "Paracetamol", "Glucose"]
            one_hot_vector = [1 if v.lower() in e_text.lower() else 0 for v in vocab_space]
            entity_data["One_Hot_Encoded"] = one_hot_vector
            
        formatted_entities.append(entity_data)

    # MỤC 2.4.2: XỬ LÝ GIÁ TRỊ KHUYẾT THIẾU ĐỘNG (Dynamic Missing Values)
    # Chỉ kỳ vọng các trường hợp lý theo từng loại tài liệu
    if document_type == "Đơn thuốc":
        doc_expected_fields = ["Patient_Name", "Diagnosis", "Medication", "Dosage"]
    elif document_type == "KQ xét nghiệm":
        doc_expected_fields = ["Patient_Name", "Diagnosis", "Lab_Value"]
    else: # Hồ sơ bệnh án
        doc_expected_fields = ["Patient_Name", "Diagnosis"]

    # Gán giá trị null để bảo vệ chỉ số Precision thay vì dự đoán bừa
    for expected in doc_expected_fields:
        if expected not in found_types:
            formatted_entities.append({
                "field": field_mapping[expected],
                "value": None  # Gán giá trị rỗng (JSON null) cho nhãn bị khuyết
            })

    # MỤC 2.4.4: ĐÓNG GÓI VÀ CẤU TRÚC HÓA DỮ LIỆU ĐẦU RA ĐỊNH DẠNG JSON
    result_json = {
        "document_type": document_type,
        "entities": formatted_entities
    }
    
    output_path = os.path.splitext(img_path)[0] + '_extracted.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False để hiển thị tiếng Việt, xuất file JSON chuẩn cấu trúc Bảng 2.1
        json.dump(result_json, f, ensure_ascii=False, indent=4)
        
    print(f"\nKết quả trích xuất chuẩn hóa đã lưu tại: {output_path}")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
    
    return result_json

if __name__ == "__main__":
    TEST_IMAGE_PATH = str(PROJECT_ROOT / "assets" / "tải_xuống.jpg")
    if os.path.exists(TEST_IMAGE_PATH):
        inference_pipeline(TEST_IMAGE_PATH)
    else:
        print(f"File {TEST_IMAGE_PATH} không tồn tại. Vui lòng cung cấp một ảnh thật để test.")
    print("\nHoàn tất bước 6: Inference Pipeline")