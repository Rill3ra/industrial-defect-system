# src/data/expose_dataset.py

from pathlib import Path

ROOT = Path("data/raw/mvtec_ad")

categoties = [
    "screw",
    "metal_nut"
]

for category in categoties:
    print(f"\n{'=' * 50}")
    print(category.upper())
    print('=' * 50)

    train_good = list(
        (ROOT / category / "train" / "good").glob("*.*")
    )

    print(f"Train good: {len(train_good)}")

    test_root = ROOT / category / "test"

    total_test = 0

    for defect_type in sorted(test_root.iterdir()):

        images = list(defect_type.glob("*.*"))

        print(
            f"{defect_type.name:20s} "
            f"{len(images):4d}"
        )

        total_test += len(images)

    print(f"Total test: {total_test}")