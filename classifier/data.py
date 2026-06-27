import os
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMG_EXTENSIONS


def get_transform(normalize: bool = False):
    if normalize:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


class FlatImageNetWithLabels(Dataset):
    def __init__(self, image_dir: str, labels_path: str, transform=None):
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.transform = transform

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not os.path.isfile(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        self.image_paths = self._load_image_paths()
        self.labels = self._load_labels()

        if len(self.image_paths) != len(self.labels):
            raise ValueError(
                f"Images and labels do not match: "
                f"{len(self.image_paths)} images vs {len(self.labels)} labels"
            )

        print("Loaded ImageNet flat dataset")
        print(f"  Image dir: {self.image_dir}")
        print(f"  Labels:    {self.labels_path}")
        print(f"  Samples:   {len(self.image_paths)}")
        print(
            f"  Label range after 1-based -> 0-based conversion: "
            f"{self.labels.min().item()} to {self.labels.max().item()}"
        )
        print("  First 5 samples:")
        for i in range(min(5, len(self.image_paths))):
            print(f"    {os.path.basename(self.image_paths[i])} -> {self.labels[i].item()}")

    def _load_image_paths(self) -> List[str]:
        image_files = [
            filename
            for filename in os.listdir(self.image_dir)
            if is_image_file(filename)
        ]
        image_files = sorted(image_files)

        if len(image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        return [os.path.join(self.image_dir, filename) for filename in image_files]

    def _load_labels(self) -> torch.Tensor:
        labels = []

        with open(self.labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                # if labels are 1-based, so convert them to 0-based.
                labels.append(int(line) - 1)

        if len(labels) == 0:
            raise RuntimeError(f"No labels found in {self.labels_path}")

        labels = torch.tensor(labels, dtype=torch.long)

        if labels.min().item() < 0:
            raise ValueError(
                "A label became negative after 1-based to 0-based conversion. "
                "Maybe your labels are already 0-based."
            )

        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataset(name, data_dir: Optional[str] = None, normalize: bool = False):
    """
    ImageNet dataset loader.

    Args:
        name: Dataset name (only "imagenet" is supported).
        data_dir: Path to the ImageNet validation image directory.
        normalize: Whether to apply ImageNet normalization.

    Labels file (labels.txt) is expected in the parent directory of data_dir.
    """

    if name.lower() != "imagenet":
        raise ValueError("Only dataset='imagenet' is supported in this data.py")

    if data_dir is None:
        raise ValueError("data_dir must be provided.")

    image_dir = data_dir
    labels_path = os.path.join(os.path.dirname(image_dir), "labels.txt")
    transform = get_transform(normalize=normalize)

    return FlatImageNetWithLabels(
        image_dir=image_dir,
        labels_path=labels_path,
        transform=transform,
    )


def make_subset(dataset, num_samples: int, seed: int, indices_path: Optional[str] = None):
    if num_samples <= 0:
        print("Using full dataset.")
        return dataset, None

    if indices_path is not None and os.path.exists(indices_path):
        indices = torch.load(indices_path, map_location="cpu")
        print(f"Loaded subset indices from {indices_path}")
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples]

        if indices_path is not None:
            os.makedirs(os.path.dirname(indices_path), exist_ok=True)
            torch.save(indices, indices_path)
            print(f"Saved subset indices to {indices_path}")

    print(f"Using subset of {len(indices)} samples.")
    return Subset(dataset, indices.tolist()), indices


def get_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=4,
    one_batch=False,
    pin_memory=True,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if one_batch:
        return iter([next(iter(loader))])

    return loader


def get_loader(dataset, batch_size, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )