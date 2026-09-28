"""Cache a pretrained classifier's pre-readout features and logits on a labelled image set. Writes
``$CACHE_DIR/vision/{model}/{dataset}.npz`` with ``features`` (N, d), ``logits`` (N, C), ``labels`` (N,) and
``index`` (N,): the 0-based validation number for ImageNet (ReaL's order), the test index for CIFAR-10, the row
number for ImageNet-16H (which also stores ``names`` and ``noise`` per row; labels are the 16-class category).
Run: ``uv run python -m src.notebooks.build_vision imagenet resnet50`` (datasets: imagenet, cifar10, imagenet16h;
models: resnet50, vit_b_16, cifar_resnet56)."""
import glob
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.code.vision import IMAGENET16, imagenet_val, penultimate

SHARDS = "/mnt/polished-lake/artifacts/public/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/validation-*.parquet"


def torchvision_model(name):
    """``(model, transform, head)`` for a torchvision ImageNet classifier with its published weights."""
    ctor, weights, head = {"resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
                           "vit_b_16": (torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1, "heads")}[name]
    return ctor(weights=weights).eval(), weights.transforms(), head


def cifar_model():
    """``(model, transform, head)``: chenyaofo's CIFAR-10 ResNet-56 from torch.hub (94.4% test accuracy) with
    the CIFAR-10 normalisation it was trained under."""
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True).eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    return model, transform, "fc"


class Images(Dataset):
    """Decoded, transformed images from ``(index, label, bytes-or-path-or-PIL)`` rows."""
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        idx, label, img = self.rows[i]
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        elif isinstance(img, (str, Path)):
            img = Image.open(img)
        return self.transform(img.convert("RGB")), label, idx


def imagenet_rows():
    return list(imagenet_val(sorted(glob.glob(SHARDS))))


def cifar10_rows(root):
    ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return [(i, int(label), img) for i, (img, label) in enumerate(ds)]


VERSIONS = {"original": 0, "phase_noise_080": 80, "phase_noise_095": 95, "phase_noise_110": 110, "phase_noise_125": 125}


def imagenet16h_rows(root):
    """ImageNet-16H: 1,200 images in five versions (original and four phase-noise levels), labelled with their
    16-class category index from the human trial CSV. Returns rows and the per-row ``(name, noise)``."""
    import csv
    categories = list(IMAGENET16)
    label = {r["image_name"]: categories.index(r["image_category"])
             for r in csv.DictReader(open(root / "human_only_classification_6per_img_export.csv"))}
    rows, meta = [], []
    for version, noise in VERSIONS.items():
        for path in sorted(glob.glob(str(root / version / "**" / "*.png"), recursive=True)):
            name = Path(path).stem
            rows.append((len(rows), label[name], path)); meta.append((name, noise))
    return rows, meta


if __name__ == "__main__":
    dataset, model_name = sys.argv[1], sys.argv[2]
    cache = Path(os.environ.get("CACHE_DIR", "outputs")) / "vision"
    out = cache / model_name; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    model, transform, head = cifar_model() if model_name == "cifar_resnet56" else torchvision_model(model_name)
    meta = None
    if dataset == "imagenet":
        rows = imagenet_rows()
    elif dataset == "cifar10":
        rows = cifar10_rows(cache / "cifar10")
    else:
        rows, meta = imagenet16h_rows(cache / "imagenet16h")
    print(f"{len(rows):,} images loaded: {time.time() - t:.0f}s", flush=True)
    loader = DataLoader(Images(rows, transform), batch_size=256, num_workers=16)
    feats, logits, labels, index = [], [], [], []
    for x, y, i in loader:
        f, l = penultimate(model, head, x, batch=256)
        feats.append(f.astype(np.float16)); logits.append(l.astype(np.float16)); labels.append(y.numpy()); index.append(i.numpy())
    feats, logits, labels, index = (np.concatenate(a) for a in (feats, logits, labels, index))
    extra = {} if meta is None else {"names": np.array([m[0] for m in meta])[index], "noise": np.array([m[1] for m in meta])[index]}
    np.savez(out / f"{dataset}.npz", features=feats, logits=logits, labels=labels, index=index, **extra)
    print(f"{model_name} on {dataset}: features {feats.shape}, top-1 {(logits.argmax(1) == labels).mean():.4f}: {time.time() - t:.0f}s")
