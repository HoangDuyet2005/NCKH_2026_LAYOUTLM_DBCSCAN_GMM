import json
import os
from pathlib import Path

# Xác định thư mục gốc dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

export_file = str(PROJECT_ROOT / "data" / "label_studio_export.json")

if not os.path.exists(export_file):
    print(f"Không tìm thấy file: {export_file}")
else:
    with open(export_file, encoding='utf-8') as f:
        data = json.load(f)

    print(f"Tong so task: {len(data)}")

    annotated = [t for t in data if t.get('annotations') and len(t['annotations']) > 0]
    print(f"So task da gan nhan: {len(annotated)}")

    labels = set()
    total_regions = 0
    for t in annotated:
        ann = t['annotations'][0]
        for r in ann.get('result', []):
            if r.get('type') == 'rectanglelabels':
                labels.update(r['value'].get('rectanglelabels', []))
                total_regions += 1

    print(f"Tong so vung da gan nhan: {total_regions}")
    print(f"Cac nhan: {labels}")

    # Count per label
    label_counts = {}
    for t in annotated:
        ann = t['annotations'][0]
        for r in ann.get('result', []):
            if r.get('type') == 'rectanglelabels':
                for lbl in r['value'].get('rectanglelabels', []):
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("\nSo luong theo nhan:")
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl}: {count}")
