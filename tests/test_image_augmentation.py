import numpy as np

from src.data.augmentation import augment_images


def test_augmentation_preserves_image_batch_shape_and_range():
    images = np.linspace(0.0, 1.0, 2 * 4 * 4 * 3, dtype=np.float32).reshape(
        2, 4, 4, 3
    )

    augmented = augment_images(
        images,
        horizontal_flip=True,
        max_shift=1,
        seed=0,
    )

    assert augmented.shape == images.shape
    assert augmented.dtype == np.float32
    assert np.all((augmented >= 0.0) & (augmented <= 1.0))


def test_augmentation_does_not_modify_original_images():
    images = np.zeros((1, 4, 4, 3), dtype=np.float32)
    images[:, 1:3, 1:3] = 1.0
    original = images.copy()

    augment_images(images, horizontal_flip=True, max_shift=1, seed=1)

    np.testing.assert_array_equal(images, original)
