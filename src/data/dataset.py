# src/data/dataser.py
import json
from PIL import Image
from torch.utils.data import Dataset


class DefectDataset(Dataset):

    def __init__(self, json_path, transform=None, only_normal=False):

        with open(json_path, "r") as f:
            samples = json.load(f)

        # сохраняем флаг
        self.only_normal = only_normal

        # фильтрация происходит ОДИН РАЗ
        if only_normal:
            samples = [s for s in samples if s["label"] == 0]

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(sample["path"]).convert("RGB")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "category": sample["category"],
            "defect_type": sample["defect_type"]
        }