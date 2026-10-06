#!/usr/bin/env python3
"""Generate five offline augmentations for the 6,237 original training images.

Run from an environment containing NumPy and Pillow:
    python augment_train.py

Verified environment: Python 3.13.9, NumPy 2.4.6, Pillow 12.2.0.
The dataset root is datasets/ beside this script. Each original image produces
one motion_blur, horizontal_flip, random_rotation, gaussian_noise, and brightness
variant under datasets/images/train_augmented/<method>/, with matching YOLO
labels under datasets/labels/train_augmented/<method>/. Filenames are
<source_id>__<method>.png and <source_id>__<method>.txt; there are no part folders.

Motion blur uses a uniformly selected 3/5/7-pixel kernel and horizontal,
vertical, or either diagonal direction. Rotation is uniform in [-15, 15]
degrees, with scale-to-fit, bicubic interpolation, and RGB (114, 114, 114)
padding. Gaussian noise has mean zero and a uniformly selected sigma in [3, 10]
on the 0..255 pixel scale. Brightness is multiplied by a uniform factor in
[0.8, 1.2]. Photometric operations round and clip to 0..255. The master seed is
42; independent per-image/per-method seeds use SHA256 and NumPy PCG64.

Only train is augmented. Original images and labels are read-only. Existing
train_augmented directories cause an error before any output is written.
Successful generation adds 31,185 images and 67,730 boxes, giving 37,422
training images and 81,276 boxes including the originals. Checks and progress
are printed to the terminal; no auxiliary metadata directory is required.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import itertools
import logging
import math
from pathlib import Path
import platform
import time

LOGGER = logging.getLogger("gmed_yolo.augmentation")

try:
    import numpy as np
    import PIL
    from PIL import Image
except ImportError:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [DEBUG] %(name)s %(levelname)s: %(message)s")
    LOGGER.exception(
        "Missing NumPy or Pillow. Install in the same Python environment with: "
        "python -m pip install numpy==2.4.6 Pillow==12.2.0"
    )
    raise SystemExit(1)


METHODS = ("motion_blur", "horizontal_flip", "random_rotation", "gaussian_noise", "brightness")
MASTER_SEED = 42


def augment_image(image, boxes, method, seed):
    labels = np.array(boxes, dtype=float, copy=True)
    rng = np.random.default_rng(seed)
    if method == "horizontal_flip":
        labels[:, 1] = 1.0 - labels[:, 1]
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), labels, {}
    if method == "brightness":
        factor = float(rng.uniform(0.8, 1.2))
        pixels = np.asarray(image).astype(np.float32)
        pixels = np.clip(np.rint(pixels * factor), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels), labels, {"factor": factor}
    if method == "motion_blur":
        kernel_size = int(rng.choice((3, 5, 7)))
        direction = str(rng.choice((
            "horizontal", "vertical", "descending_diagonal", "ascending_diagonal",
        )))
        pixels = np.asarray(image).astype(np.float32)
        height, width = pixels.shape[:2]
        radius = kernel_size // 2
        padded = np.pad(pixels, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        blurred = np.zeros_like(pixels)
        for offset in range(-radius, radius + 1):
            dy = 0 if direction == "horizontal" else offset
            dx = 0 if direction == "vertical" else offset
            if direction == "ascending_diagonal":
                dx = -offset
            blurred += padded[
                radius + dy:radius + dy + height, radius + dx:radius + dx + width,
            ]
        blurred = np.rint(blurred / kernel_size).astype(np.uint8)
        parameters = {"kernel_size": kernel_size, "direction": direction}
        return Image.fromarray(blurred), labels, parameters
    if method == "gaussian_noise":
        sigma = float(rng.uniform(3, 10))
        pixels = np.asarray(image).astype(np.float32)
        noise = rng.normal(0, sigma, size=pixels.shape).astype(np.float32)
        pixels = np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels), labels, {"sigma": sigma}
    if method == "random_rotation":
        angle = float(rng.uniform(-15, 15))
        output, labels = rotate_keep_all(image, boxes, angle)
        c, s = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
        scale = min(image.width / (c * image.width + s * image.height),
                    image.height / (s * image.width + c * image.height))
        parameters = {"angle_degrees": angle, "scale": scale, "padding_rgb": [114, 114, 114]}
        return output, labels, parameters
    raise ValueError(f"Unknown augmentation method: {method}")


def rotate_keep_all(image, boxes, angle_degrees):
    width, height = image.size
    radians = math.radians(angle_degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    # Exact quarter turns avoid interpolation caused by trigonometric roundoff.
    if abs(cosine) < 1e-12:
        cosine = 0.0
    if abs(sine) < 1e-12:
        sine = 0.0
    scale = min(
        width / (abs(cosine) * width + abs(sine) * height),
        height / (abs(sine) * width + abs(cosine) * height),
    )
    a, b = scale * cosine, scale * sine
    matrix = np.array([
        [a, b, width / 2 - a * width / 2 - b * height / 2],
        [-b, a, height / 2 + b * width / 2 - a * height / 2],
        [0, 0, 1],
    ])
    inverse = np.linalg.inv(matrix)
    output = image.transform(
        image.size, Image.Transform.AFFINE, tuple(inverse[:2].ravel()),
        resample=Image.Resampling.BICUBIC, fillcolor=(114, 114, 114),
    )
    labels = np.array(boxes, dtype=float, copy=True)
    for index, (_, cx, cy, bw, bh) in enumerate(boxes):
        left, right = (cx - bw / 2) * width, (cx + bw / 2) * width
        top, bottom = (cy - bh / 2) * height, (cy + bh / 2) * height
        corners = np.array([
            [left, top, 1], [right, top, 1], [right, bottom, 1], [left, bottom, 1],
        ])
        transformed = (corners @ matrix.T)[:, :2]
        low, high = transformed.min(axis=0), transformed.max(axis=0)
        labels[index, 1:] = [
            (low[0] + high[0]) / (2 * width), (low[1] + high[1]) / (2 * height),
            (high[0] - low[0]) / width, (high[1] - low[1]) / height,
        ]
    return output, labels


def save_variant(image, boxes, method, source_label, image_path, label_path):
    with image_path.open("xb") as stream:
        image.save(stream, format="PNG", compress_level=3)
    with label_path.open("xb") as stream:
        if method in ("horizontal_flip", "random_rotation"):
            lines = [
                f"{int(row[0])} " + " ".join(f"{value:.9f}" for value in row[1:])
                for row in boxes
            ]
            stream.write(("\n".join(lines) + "\n").encode("ascii"))
        else:
            stream.write(source_label.read_bytes())


def check_boxes(boxes, image_size, context):
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not len(boxes) or not np.isfinite(boxes).all():
        raise ValueError(f"{context}: expected nonempty finite N×5 YOLO labels")
    if not np.isin(boxes[:, 0], [0, 1, 2, 3]).all() or (boxes[:, 3:5] <= 0).any():
        raise ValueError(f"{context}: invalid class IDs or nonpositive box dimensions")
    size = np.array(image_size)
    lower = (boxes[:, 1:3] - boxes[:, 3:5] / 2) * size
    upper = (boxes[:, 1:3] + boxes[:, 3:5] / 2) * size
    # Two original labels have subpixel rounding excursions; preserve their bytes.
    if (lower < -0.001).any() or (upper > size + 0.001).any():
        raise ValueError(f"{context}: a box extends beyond the 0.001-pixel edge tolerance")


def process_sample(root, source_id):
    try:
        image_path = root / "datasets/images/train" / f"{source_id}.jpg"
        label_path = root / "datasets/labels/train" / f"{source_id}.txt"
        source_image_bytes, source_label_bytes = image_path.read_bytes(), label_path.read_bytes()
        with Image.open(io.BytesIO(source_image_bytes)) as source:
            image = source.convert("RGB")
        boxes = np.loadtxt(io.BytesIO(source_label_bytes), ndmin=2)
        check_boxes(boxes, image.size, f"source={source_id}")
        outputs = []
        for method in METHODS:
            # Per-sample seeds make results independent of worker scheduling.
            seed_text = f"{MASTER_SEED}:{source_id}:{method}".encode("ascii")
            seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
            output, labels, parameters = augment_image(image, boxes, method, seed)
            check_boxes(labels, image.size, f"source={source_id} method={method}")
            if output.size != image.size or not np.array_equal(labels[:, 0], boxes[:, 0]):
                raise ValueError(
                    f"source={source_id} method={method}: size/classes/count/order changed"
                )
            stem = f"{source_id}__{method}"
            output_image = root / "datasets/images/train_augmented" / method / f"{stem}.png"
            output_label = root / "datasets/labels/train_augmented" / method / f"{stem}.txt"
            save_variant(output, labels, method, label_path, output_image, output_label)
            with Image.open(output_image) as saved:
                np.testing.assert_array_equal(np.asarray(saved), np.asarray(output))
            saved_labels = np.loadtxt(output_label, ndmin=2)
            check_boxes(saved_labels, image.size, f"saved={stem}")
            np.testing.assert_allclose(saved_labels, labels, rtol=0, atol=5.1e-10)
            outputs.append({"method": method, "seed": seed, "parameters": parameters,
                            "boxes": len(saved_labels)})
        return {"source_id": source_id, "outputs": outputs}
    except Exception:
        LOGGER.exception("[DEBUG] process_sample failed source_id=%s root=%s", source_id, root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    output_folders = [root / "datasets" / kind / "train_augmented" for kind in ("images", "labels")]
    for folder in output_folders:
        if folder.exists() or folder.is_symlink():
            raise FileExistsError(
                f"Existing augmentation output: {folder}. Refusing to overwrite it. "
                "Generate only in a clean copy containing the original dataset."
            )
    sources = sorted((root / "datasets/images/train").glob("*.jpg"),
                     key=lambda path: int(path.stem))
    labels = list((root / "datasets/labels/train").glob("*.txt"))
    if len(sources) != 6237 or {p.stem for p in sources} != {p.stem for p in labels}:
        raise ValueError("Expected exactly 6,237 matching original train image/label pairs")
    class_counts = np.zeros(4, dtype=int)
    for path in sources:
        boxes = np.loadtxt(root / "datasets/labels/train" / f"{path.stem}.txt", ndmin=2)
        with Image.open(path) as image:
            check_boxes(boxes, image.size, f"preflight={path.stem}")
        class_counts += np.bincount(boxes[:, 0].astype(int), minlength=4)
    if class_counts.tolist() != [6237, 2147, 1328, 3834]:
        raise ValueError(
            f"Original train class box totals differ from expected counts: {class_counts.tolist()}"
        )

    for folder in output_folders:
        for method in METHODS:
            (folder / method).mkdir(parents=True)
    LOGGER.info("[DEBUG] environment python=%s numpy=%s Pillow=%s master_seed=%s",
                platform.python_version(), np.__version__, PIL.__version__, MASTER_SEED)
    LOGGER.info("[DEBUG] starting sources=%s methods=%s workers=8 original_boxes=%s",
                len(sources), len(METHODS), int(class_counts.sum()))
    totals = {method: [0, 0] for method in METHODS}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_sample, itertools.repeat(root),
                               (p.stem for p in sources), chunksize=1)
        for completed, record in enumerate(results, start=1):
            for output in record["outputs"]:
                totals[output["method"]][0] += 1
                totals[output["method"]][1] += output["boxes"]
            if completed % 100 == 0 or completed == len(sources):
                LOGGER.info(
                    "[DEBUG] completed_sources=%s/%s generated_pairs=%s elapsed_seconds=%.1f",
                    completed, len(sources), completed * 5, time.monotonic() - started,
                )
    for method, counts in totals.items():
        if counts != [6237, 13546]:
            raise RuntimeError(f"Unexpected output totals for {method}: images/boxes={counts}")
        actual_images = list((output_folders[0] / method).glob("*.png"))
        actual_labels = list((output_folders[1] / method).glob("*.txt"))
        image_ids, label_ids = {p.stem for p in actual_images}, {p.stem for p in actual_labels}
        if len(actual_images) != 6237 or image_ids != label_ids:
            raise RuntimeError(f"Output image/label pairs do not match for {method}")
        LOGGER.info("[DEBUG] method=%s images=%s boxes=%s", method, *counts)
    LOGGER.info("[DEBUG] complete added_images=31185 added_boxes=67730 "
                "total_train_images=37422 total_train_boxes=81276")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [DEBUG] %(name)s %(levelname)s: %(message)s")
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("[DEBUG] Training augmentation failed")
        raise SystemExit(1)
