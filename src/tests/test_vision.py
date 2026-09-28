"""Vision classifiers: soft label tables from human relabellings, and pre-readout features of pretrained
models on the labelled images."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.code.vision import imagenet_val, penultimate, real_table, soft_table

CACHE = Path(os.environ.get("CACHE_DIR", "outputs"))


def test_real_table_splits_each_images_votes_over_its_valid_labels():
    """Images of class x vote 1/|R| for each label in their ReaL set; an empty set casts no vote."""
    labels = [0, 0, 1, 1, 2]
    real_sets = [[0], [0, 1], [1], [], [2, 0]]
    T = real_table(real_sets, labels, 3)
    assert np.allclose(T, [[1.5, 0.5, 0], [0, 1, 0], [0.5, 0, 0.5]])


def test_soft_table_averages_the_annotator_distribution_per_class():
    """Rows are the mean human label distribution over the images of each class, so they sum to one."""
    probs = np.array([[0.8, 0.2, 0], [0.6, 0.4, 0], [0, 0, 1.0]])
    T = soft_table(probs, [0, 0, 1], 2)
    assert np.allclose(T, [[0.7, 0.3, 0], [0, 0, 1.0]]) and np.allclose(T.sum(1), 1)


def test_penultimate_returns_the_features_the_head_maps_to_the_logits():
    """The head is bypassed for the features and applied afterwards for the logits, so the two are
    related by the model's own readout; works for a ResNet (``fc``) and a ViT (``heads``)."""
    import torchvision
    torch.manual_seed(0)
    x = torch.randn(3, 3, 64, 64)
    model = torchvision.models.resnet18(weights=None, num_classes=5).eval()
    fc = model.fc
    features, logits = penultimate(model, "fc", x, device="cpu", batch=2)
    assert features.shape == (3, 512) and logits.shape == (3, 5)
    with torch.no_grad():
        assert np.allclose(logits, fc(torch.as_tensor(features)).numpy(), atol=1e-5)
    vit = torchvision.models.vit_b_16(weights=None, num_classes=5, image_size=64).eval()
    heads = vit.heads
    f2, l2 = penultimate(vit, "heads", x, device="cpu", batch=2)
    with torch.no_grad():
        assert f2.shape == (3, 768) and np.allclose(l2, heads(torch.as_tensor(f2)).numpy(), atol=1e-5)


def test_imagenet_val_reads_index_label_and_bytes_from_the_parquet_path(tmp_path):
    """Each row's ``image.path`` names the validation image; its number (1-based) becomes the ReaL index."""
    import pyarrow as pa, pyarrow.parquet as pq
    rows = pa.table({"image": [{"bytes": b"a", "path": "ILSVRC2012_val_00000007_n01440764.JPEG"},
                               {"bytes": b"b", "path": "ILSVRC2012_val_00050000_n15075141.JPEG"}],
                     "label": [0, 999]})
    pq.write_table(rows, tmp_path / "validation-00000-of-00001.parquet")
    out = list(imagenet_val([tmp_path / "validation-00000-of-00001.parquet"]))
    assert out == [(6, 0, b"a"), (49999, 999, b"b")]


@pytest.mark.slow
def test_real_labels_share_the_class_order_of_the_cached_validation_set():
    """Label-order recovery: with matching class ids, the original label sits inside the ReaL set for
    the large majority of non-empty images; a mismatched ordering would give almost none."""
    import glob
    shards = sorted(glob.glob("/mnt/polished-lake/artifacts/public/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/validation-*.parquet"))
    real = json.load(open(CACHE / "vision" / "labels" / "real.json"))
    hits, n = 0, 0
    for i, label, _ in imagenet_val(shards, columns=("label",)):
        if real[i]:
            n += 1; hits += label in real[i]
    assert n == 50000 - 3163 and hits / n > 0.8


@pytest.mark.slow
def test_resnet50_reproduces_its_reported_top1_on_a_validation_subset():
    """The preprocessing and head bypass are right when torchvision's ResNet-50 V2 scores its
    published 80.9% top-1 (within the sampling error of a 5k-image subset)."""
    import glob, io
    import torchvision
    from PIL import Image
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights).eval()
    tf = weights.transforms()
    shards = sorted(glob.glob("/mnt/polished-lake/artifacts/public/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/validation-*.parquet"))[:2]
    rows = [(label, tf(Image.open(io.BytesIO(b)).convert("RGB"))) for _, label, b in imagenet_val(shards)]
    rows = rows[:5000]
    _, logits = penultimate(model, "fc", torch.stack([r[1] for r in rows]), batch=256)
    top1 = (logits.argmax(1) == np.array([r[0] for r in rows])).mean()
    assert abs(top1 - 0.809) < 0.02, top1
