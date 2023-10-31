import colorsys
import glob
import os

import h5py
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import zoom
from skimage.io import imsave
from sklearn.decomposition import PCA


def visualize(name):
    with h5py.File(os.path.join("features", name)) as fh:
        data = fh["data"][()]
        features = fh["F8"][()]
        scale = data.shape[1] / features.shape[1]

    shape = features.shape[:-1]
    features = features.reshape(-1, features.shape[-1])
    transformer = PCA(n_components=8)
    vec = transformer.fit_transform(features)

    alpha = 0
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    hues = ((alpha + angles + np.pi) % (2 * np.pi)) / (2 * np.pi)

    H = hues
    S = np.ones_like(H) * 0.9
    V = np.ones_like(H) * 0.75

    vec = np.stack([H, S, V], axis=-1)
    vec = hsv_to_rgb(vec)

    vec = vec.reshape(*shape, 3)
    # vec = V.reshape(shape)
    vec = (255 * vec).astype(np.uint8)
    vec = zoom(vec, zoom=(1, scale, scale, 1), order=0)

    for i, v in enumerate(vec):
        if i % 20 == 0:
            imsave(f"slice_{i:03}.png", v[::-1])


if __name__ == "__main__":
    for name in list(sorted(glob.glob("*.hdf", root_dir="features")))[:1]:
        visualize(name)
