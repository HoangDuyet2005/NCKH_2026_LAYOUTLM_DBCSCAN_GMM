import json

with open('project-9-at-2026-05-24-18-15-d6ecab75.json', encoding='utf-8') as f:
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
