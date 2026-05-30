import os
import json
import glob
from PIL import Image
from tqdm import tqdm

# Định nghĩa hệ nhãn BIO
LABELS = [
    "O",
    "B-Patient_Name", "I-Patient_Name",
    "B-Diagnosis",    "I-Diagnosis",
    "B-Medication",   "I-Medication",
    "B-Dosage",       "I-Dosage",
    "B-Lab_Value",    "I-Lab_Value"
]
label2id = {label: i for i, label in enumerate(LABELS)}


def normalize_bbox(bbox, width, height):
    """Chuẩn hóa bounding box về tỷ lệ [0, 1000] cho LayoutLMv3."""
    x_min, y_min, x_max, y_max = bbox
    return [
        max(0, min(1000, int(1000 * (x_min / width)))),
        max(0, min(1000, int(1000 * (y_min / height)))),
        max(0, min(1000, int(1000 * (x_max / width)))),
        max(0, min(1000, int(1000 * (y_max / height))))
    ]


def convert_label_studio_export(
    export_json_path: str,
    dataset_dir: str,
    output_json_path: str = "training_data.json"
):
    """
    Đọc file JSON xuất từ Label Studio và chuyển đổi sang định dạng
    chuẩn để huấn luyện LayoutLMv3 (Step 5).

    Định dạng export của Label Studio (dạng JSON):
    [
      {
        "data": {"image": "/path/to/image.jpg"},
        "annotations": [{
          "result": [
            {
              "type": "rectanglelabels",
              "value": {
                "x": ..., "y": ..., "width": ..., "height": ...,
                "labels": ["Patient_Name"]
              },
              "origin": "manual"
            },
            ...
          ]
        }]
      },
      ...
    ]
    """
    print(f"Đang đọc file export của Label Studio: {export_json_path}")
    with open(export_json_path, 'r', encoding='utf-8') as f:
        ls_data = json.load(f)

    training_records = []

    for task in tqdm(ls_data, desc="Đang chuyển đổi nhãn Label Studio"):
        try:
            # Lấy đường dẫn ảnh từ trường data
            image_path_raw = task["data"].get("image", "")

            # Xử lý các loại đường dẫn (từ HTTP server hoặc local-files)
            if "http://localhost:8081/" in image_path_raw:
                img_path = image_path_raw.split("http://localhost:8081/")[-1]
            elif "?d=" in image_path_raw:
                img_path = image_path_raw.split("?d=")[-1]
            else:
                img_path = image_path_raw
                
            # Loại bỏ query parameter (như ?t=...)
            if "?" in img_path:
                img_path = img_path.split("?")[0]

            img_path = img_path.strip()
            
            # Đảm bảo đường dẫn hợp lệ trên Windows
            img_path = os.path.normpath(img_path)

            if not os.path.exists(img_path):
                print(f"Không tìm thấy ảnh: {img_path}, bỏ qua.")
                continue

            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            # Đọc kết quả OCR đã làm sạch từ file JSON bước 2
            json_ocr_path = os.path.splitext(img_path)[0] + '.json'
            if not os.path.exists(json_ocr_path):
                print(f"Không tìm thấy OCR JSON cho: {img_path}, bỏ qua.")
                continue

            with open(json_ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            ocr_items = ocr_data.get("ocr_results_cleaned", [])
            if not ocr_items:
                continue

            # Xây dựng danh sách từ và bbox OCR
            words = [item["text"] for item in ocr_items]
            raw_bboxes = [item["bbox"] for item in ocr_items]  # pixel coords

            # Mặc định tất cả token là "O"
            ner_tags = ["O"] * len(words)

            # Đọc các annotation từ Label Studio
            annotations = task.get("annotations", [])
            if not annotations:
                # Không có annotation → bỏ qua ảnh này hoặc giữ toàn "O"
                # Trong nghiên cứu: chỉ lấy những ảnh đã được gán nhãn
                continue

            labeled_results = annotations[0].get("result", [])

            for result in labeled_results:
                # Chỉ xử lý loại kết quả "rectanglelabels"
                if result.get("type") != "rectanglelabels":
                    continue

                value = result.get("value", {})
                label_names = value.get("rectanglelabels", [])
                if not label_names:
                    label_names = value.get("labels", [])
                if not label_names:
                    continue

                entity_label = label_names[0]  # Lấy nhãn đầu tiên

                # Tọa độ bbox trong Label Studio tính theo % (0-100)
                x_pct = value.get("x", 0)
                y_pct = value.get("y", 0)
                w_pct = value.get("width", 0)
                h_pct = value.get("height", 0)

                # Chuyển về pixel
                orig_w = result.get("original_width", width)
                orig_h = result.get("original_height", height)

                ann_x_min = (x_pct / 100.0) * orig_w
                ann_y_min = (y_pct / 100.0) * orig_h
                ann_x_max = ann_x_min + (w_pct / 100.0) * orig_w
                ann_y_max = ann_y_min + (h_pct / 100.0) * orig_h

                # Gán nhãn BIO cho từng OCR token chồng lấp với annotation bbox
                first_token_in_entity = True
                for idx, (word, bbox) in enumerate(zip(words, raw_bboxes)):
                    tok_x_min, tok_y_min, tok_x_max, tok_y_max = bbox

                    # Tính giao nhau (IoU đơn giản: chồng lấp theo diện tích)
                    inter_x_min = max(tok_x_min, ann_x_min)
                    inter_y_min = max(tok_y_min, ann_y_min)
                    inter_x_max = min(tok_x_max, ann_x_max)
                    inter_y_max = min(tok_y_max, ann_y_max)

                    inter_area = max(0, inter_x_max - inter_x_min) * \
                                 max(0, inter_y_max - inter_y_min)
                    tok_area = max(1, (tok_x_max - tok_x_min) *
                                   (tok_y_max - tok_y_min))

                    # Nếu ít nhất 50% diện tích token nằm trong annotation bbox
                    if inter_area / tok_area >= 0.5:
                        if first_token_in_entity:
                            ner_tags[idx] = f"B-{entity_label}"
                            first_token_in_entity = False
                        else:
                            ner_tags[idx] = f"I-{entity_label}"

            # Chuẩn hóa bbox về [0, 1000]
            normalized_bboxes = [
                normalize_bbox(bb, width, height) for bb in raw_bboxes
            ]

            # Chuyển nhãn chuỗi sang id
            ner_tag_ids = [
                label2id.get(tag, label2id["O"]) for tag in ner_tags
            ]

            training_records.append({
                "image_path": img_path,
                "tokens":     words,
                "bboxes":     normalized_bboxes,
                "ner_tags":   ner_tag_ids
            })

        except Exception as e:
            print(f"Lỗi khi xử lý task: {e}")
            continue

    print(f"\nTổng số mẫu huấn luyện hợp lệ: {len(training_records)}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(training_records, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu dữ liệu huấn luyện tại: {output_json_path}")
    return output_json_path


if __name__ == "__main__":
    # Đường dẫn tới file export từ Label Studio
    # (Sau khi gán nhãn xong, Export → JSON format)
    LABEL_STUDIO_EXPORT = "label_studio_export.json"
    DATASET_DIR = "dataset"
    OUTPUT_FILE = "training_data.json"

    if not os.path.exists(LABEL_STUDIO_EXPORT):
        print(
            f"[LỖI] Chưa tìm thấy file '{LABEL_STUDIO_EXPORT}'.\n"
            "Hướng dẫn:\n"
            "  1. Mở Label Studio tại http://localhost:8080\n"
            "  2. Import file 'label_studio_import.json'\n"
            "  3. Gán nhãn BIO cho từng ảnh\n"
            "  4. Export → JSON format → lưu thành 'label_studio_export.json'\n"
            "  5. Chạy lại script này"
        )
    else:
        convert_label_studio_export(
            LABEL_STUDIO_EXPORT,
            DATASET_DIR,
            OUTPUT_FILE
        )
        print("Hoàn tất bước 4b: Chuyển đổi nhãn Label Studio → dữ liệu huấn luyện.")
