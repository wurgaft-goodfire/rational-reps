"""Vision classifiers against soft label tables: class x class tables from human relabellings (ImageNet-ReaL,
CIFAR-10H), and the pre-readout features of pretrained models on the same images."""
import re

import numpy as np
import torch

_VAL = re.compile(r"ILSVRC2012_val_(\d{8})")

# Geirhos et al.'s 16-class-ImageNet: each category pools these ImageNet classes (0-999, sorted-synset order);
# the order is ImageNet-16H's. From bethgelab/model-vs-human, modelvshuman/helper/human_categories.py.
IMAGENET16 = {
    "knife": [499], "keyboard": [508, 878], "elephant": [385, 386], "bicycle": [444, 671], "airplane": [404],
    "clock": [409, 530, 892], "oven": [766], "chair": [423, 559, 765, 857], "bear": [294, 295, 296, 297],
    "boat": [472, 554, 625, 814, 914], "cat": [281, 282, 283, 284, 285, 286],
    "bottle": [440, 720, 737, 898, 899, 901, 907], "truck": [555, 569, 656, 675, 717, 734, 864, 867],
    "car": [436, 511, 817],
    "bird": [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94,
             95, 96, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143,
             144, 145],
    "dog": [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
            173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194,
            195, 196, 197, 198, 199, 200, 201, 202, 203, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
            217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
            239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255, 256, 257, 259, 261, 262, 263,
            265, 266, 267, 268]}


def superclass_logits(logits, groups):
    """(N, len(groups)): each group's logit is the log-sum-exp of its member classes' logits, so a softmax
    over groups is the summed member probabilities; ``groups`` {name: [class indices]} in the output order."""
    Z = np.asarray(logits, dtype=np.float64)
    out = np.empty((len(Z), len(groups)))
    for j, members in enumerate(groups.values()):
        m = Z[:, members].max(1, keepdims=True)
        out[:, j] = (m + np.log(np.exp(Z[:, members] - m).sum(1, keepdims=True)))[:, 0]
    return out


def vote_table(trials, categories):
    """``(items, T)``: ``items`` the distinct (image, noise level) pairs in first-seen order and ``T`` (items,
    len(categories)) the count of trials answering each category, from ``trials`` of (image, noise, answer)."""
    index = {c: j for j, c in enumerate(categories)}
    items, rows = [], {}
    for image, noise, answer in trials:
        key = (image, noise)
        if key not in rows:
            rows[key] = len(items); items.append(key)
    T = np.zeros((len(items), len(categories)))
    for image, noise, answer in trials:
        T[rows[image, noise], index[answer]] += 1
    return items, T


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
