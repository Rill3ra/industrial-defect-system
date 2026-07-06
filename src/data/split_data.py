from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import json

SEED = 42

ROOT = Path("data/raw/mvtec_ad")
OUTPUT_DIR = Path("data/splits")

CATEGORIES = [
    "screw",
    "metal_nut"
]

samples = []


def add_sample(path, label, category, defect_type):
    samples.append(
        {
            "path": str(path),
            "label": label,              # 0 = normal, 1 = defect
            "category": category,
            "defect_type": defect_type
        }
    )


for category in CATEGORIES:

    print(f"\nProcessing {category}...")

    # -----------------------
    # TRAIN GOOD
    # -----------------------
    train_good_dir = ROOT / category / "train" / "good"

    for img_path in train_good_dir.glob("*.*"):
        add_sample(
            path=img_path,
            label=0,
            category=category,
            defect_type="good"
        )

    # -----------------------
    # TEST
    # -----------------------
    test_dir = ROOT / category / "test"

    for defect_folder in sorted(test_dir.iterdir()):

        if not defect_folder.is_dir():
            continue

        defect_type = defect_folder.name

        for img_path in defect_folder.glob("*.*"):

            label = 0 if defect_type == "good" else 1

            add_sample(
                path=img_path,
                label=label,
                category=category,
                defect_type=defect_type
            )


print("\nTotal samples:", len(samples))

# ====================================================
# STRATIFICATION KEY
# ====================================================

stratify_labels = [
    f"{item['category']}_{item['label']}"
    for item in samples
]

# ====================================================
# TRAIN / TEMP
# ====================================================

train_data, temp_data = train_test_split(
    samples,
    test_size=0.30,
    random_state=SEED,
    stratify=stratify_labels
)

# ====================================================
# VAL / TEST
# ====================================================

temp_stratify = [
    f"{item['category']}_{item['label']}"
    for item in temp_data
]

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_stratify
)

# ====================================================
# SAVE
# ====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open(OUTPUT_DIR / "val.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open(OUTPUT_DIR / "test.json", "w") as f:
    json.dump(test_data, f, indent=4)

# ====================================================
# STATISTICS
# ====================================================

def print_stats(name, data):

    labels = [item["label"] for item in data]

    counter = Counter(labels)

    normal = counter[0]
    defect = counter[1]

    print(f"\n{name}")
    print("-" * 30)
    print(f"Total   : {len(data)}")
    print(f"Normal  : {normal}")
    print(f"Defect  : {defect}")


print_stats("TRAIN", train_data)
print_stats("VALIDATION", val_data)
print_stats("TEST", test_data)

print("\nFiles saved:")
print("data/splits/train.json")
print("data/splits/val.json")
print("data/splits/test.json")