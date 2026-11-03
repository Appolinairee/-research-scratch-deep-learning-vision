from __future__ import annotations

import numpy as np


def augment_images(
    images: np.ndarray,
    *,
    horizontal_flip: bool = True,
    max_shift: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply deterministic, label-preserving CIFAR-style transforms per image."""
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, height, width, channels)")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    rng = np.random.default_rng(seed)
    augmented = np.array(images, dtype=np.float32, copy=True)
    for index in range(len(augmented)):
        image = augmented[index]
        if horizontal_flip and rng.random() < 0.5:
            image = image[:, ::-1, :]
        if max_shift:
            shifted = np.zeros_like(image)
            vertical = int(rng.integers(-max_shift, max_shift + 1))
            horizontal = int(rng.integers(-max_shift, max_shift + 1))
            source_y_start = max(0, -vertical)
            source_y_end = min(image.shape[0], image.shape[0] - vertical)
            source_x_start = max(0, -horizontal)
            source_x_end = min(image.shape[1], image.shape[1] - horizontal)
            target_y_start = max(0, vertical)
            target_y_end = target_y_start + (source_y_end - source_y_start)
            target_x_start = max(0, horizontal)
            target_x_end = target_x_start + (source_x_end - source_x_start)
            shifted[target_y_start:target_y_end, target_x_start:target_x_end] = (
                image[source_y_start:source_y_end, source_x_start:source_x_end]
            )
            image = shifted
        augmented[index] = image
    return np.clip(augmented, 0.0, 1.0)
