"""Vision classifiers against soft label tables: class x class tables from human relabellings (ImageNet-ReaL,
CIFAR-10H), and the pre-readout features of pretrained models on the same images."""
import re

import numpy as np
import torch

_VAL = re.compile(r"ILSVRC2012_val_(\d{8})")


def real_table(real_sets, labels, C):
    """(C, C) votes: each image of class ``labels[i]`` gives 1/|R_i| to every label in its ReaL set
    ``R_i``; an empty set votes for nothing."""
    T = np.zeros((C, C))
    for x, R in zip(labels, real_sets):
        for y in R:
            T[x, y] += 1 / len(R)
    return T


def soft_table(probs, labels, C):
    """(C, C): the mean annotator label distribution over the images of each class."""
    probs = np.asarray(probs, dtype=np.float64); labels = np.asarray(labels)
    return np.stack([probs[labels == x].mean(0) for x in range(C)])


def imagenet_val(paths, columns=("label", "image")):
    """Rows of the cached ILSVRC-2012 validation parquet shards as ``(index, label, bytes)``: the 0-based
    validation number parsed from the file name (ReaL's order), the label, and the JPEG bytes (``None``
    unless ``"image"`` is requested)."""
    import pyarrow.parquet as pq
    cols = ["label", "image"] if "image" in columns else ["label", "image.path"]
    for path in paths:
        for b in pq.ParquetFile(path).iter_batches(batch_size=512, columns=cols):
            d = b.to_pydict()
            for label, img in zip(d["label"], d["image"]):
                yield int(_VAL.search(img["path"]).group(1)) - 1, label, img.get("bytes")


@torch.no_grad()
def penultimate(model, head, x, batch=256, device="cuda"):
    """``(features, logits)`` of a torchvision classifier on preprocessed images ``x`` (N, 3, H, W): the
    input to its head module (attribute ``head``, e.g. "fc" or "heads") and that head's output, in
    batches on ``device``. The model is left as it was."""
    head_module = getattr(model, head)
    setattr(model, head, torch.nn.Identity())
    model.to(device); head_module.to(device)
    feats, logits = [], []
    for i in range(0, len(x), batch):
        f = model(x[i:i + batch].to(device))
        feats.append(f.float().cpu()); logits.append(head_module(f).float().cpu())
    setattr(model, head, head_module)
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()
