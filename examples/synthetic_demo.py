"""Illustrate the Phase R feature and fusion math with synthetic arrays."""

import numpy as np


def patches(image):
    windows = np.lib.stride_tricks.sliding_window_view(image, (6, 6), axis=(0, 1))
    grid = windows.transpose(0, 1, 3, 4, 2).reshape(*windows.shape[:2], 108)
    centered = grid - grid.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(centered.var(axis=-1, keepdims=True) + 10.0)


def triangle(patch_grid, centroids):
    distances = np.linalg.norm(patch_grid[..., None, :] - centroids, axis=-1)
    return np.maximum(distances.mean(axis=-1, keepdims=True) - distances, 0)


def polarity(patch_grid, centroids):
    response = patch_grid @ centroids.T
    return np.concatenate(
        (np.maximum(response - 0.5, 0), np.maximum(-response - 0.5, 0)),
        axis=-1,
    )


def pool(grid):
    row, column = (size // 2 for size in grid.shape[:2])
    regions = (
        grid,
        grid[:row, :column],
        grid[row:, :column],
        grid[:row, column:],
        grid[row:, column:],
    )
    return np.concatenate(
        [stat for region in regions
         for stat in (region.mean(axis=(0, 1)), region.max(axis=(0, 1)))]
    )


def fuse(mlp, tree, fine_sgd, polarity_sgd, coarse_sgd):
    inner = 0.5 * mlp + 0.5 * tree
    phase_h = 0.5 * inner + 0.5 * fine_sgd
    phase_p = 0.75 * phase_h + 0.25 * polarity_sgd
    return 0.75 * phase_p + 0.25 * coarse_sgd


def main():
    rng = np.random.default_rng(42)
    image = rng.uniform(0, 255, (32, 32, 3))
    reduced = image.reshape(16, 2, 16, 2, 3).mean(axis=(1, 3))
    centers = rng.normal(size=(4, 108))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    fine = patches(image)
    coarse = patches(reduced)
    assert fine.shape == (27, 27, 108) and coarse.shape == (11, 11, 108)

    features = (
        pool(triangle(fine, centers)),
        pool(polarity(fine, centers)),
        np.sqrt(pool(triangle(coarse, centers))),
    )
    assert [vector.size for vector in features] == [40, 80, 40]
    assert all(np.isfinite(vector).all() and (vector >= 0).all() for vector in features)
    assert np.allclose(triangle(np.array([0.0]), np.array([[0.0], [2.0]])), [1, 0])
    assert np.allclose(polarity(np.array([2.0]), np.array([[1.0], [-1.0]])), [1.5, 0, 0, 1.5])

    probabilities = rng.dirichlet(np.ones(10), size=5)
    combined = fuse(*probabilities)
    flat_weights = np.array([9 / 64, 9 / 64, 9 / 32, 3 / 16, 1 / 4])
    assert np.allclose(combined, flat_weights @ probabilities)
    assert np.isclose(combined.sum(), 1) and (combined >= 0).all()
    print("Synthetic checks passed: three descriptors and ten-class probability fusion.")
    print("No training or validation scoring was performed.")


if __name__ == "__main__":
    main()
