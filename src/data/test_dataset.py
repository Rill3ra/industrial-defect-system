from torch.utils.data import DataLoader

from src.data.dataset import DefectDataset
from src.data.transforms import get_train_transforms


dataset = DefectDataset(
    json_path="data/splits/train.json",
    transform=get_train_transforms()
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

batch = next(iter(loader))

print(batch["image"].shape)
print(batch["label"].shape)

print(batch["category"][:5])
print(batch["defect_type"][:5])