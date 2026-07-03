import argparse
import gzip
import random
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


BRIGHTNESS_LEVELS = (1.0, 0.7, 0.4, 0.1)
SOBEL_DISPLAY_VMAX = float(np.sqrt(2.0) * 4.0 * 255.0)


def open_idx_file(path):
    path = Path(path)
    if path.exists():
        return path.open("rb")

    gz_path = Path(f"{path}.gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")

    raise FileNotFoundError(f"MNIST file not found: {path} or {gz_path}")


def load_mnist_test(root="./dataset"):
    raw_dir = Path(root) / "MNIST" / "raw"
    images_path = raw_dir / "t10k-images-idx3-ubyte"
    labels_path = raw_dir / "t10k-labels-idx1-ubyte"

    with open_idx_file(images_path) as image_file:
        magic, count, rows, cols = struct.unpack(">IIII", image_file.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {images_path}")
        images = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(count, rows, cols)

    with open_idx_file(labels_path) as label_file:
        magic, count = struct.unpack(">II", label_file.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {labels_path}")
        labels = np.frombuffer(label_file.read(), dtype=np.uint8)

    if len(images) != len(labels):
        raise ValueError(f"Image/label count mismatch: {len(images)} images, {len(labels)} labels.")

    return images, labels


def apply_brightness(image, brightness):
    bright_image = image.astype(np.float32) * brightness
    return np.clip(bright_image, 0, 255).astype(np.float32)


def convolve2d(image, kernel):
    image = image.astype(np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            region = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            output[row, col] = np.sum(region * kernel)

    return output


def sobel_edge(image):
    if cv2 is not None:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d(image, kernel_x)
    gy = convolve2d(image, kernel_y)
    return np.sqrt(gx**2 + gy**2)


def to_transparent_rgba(image, vmax):
    intensity = np.clip(image.astype(np.float32) / vmax, 0.0, 1.0)
    rgba = np.zeros((*image.shape, 4), dtype=np.float32)
    rgba[..., :3] = 1.0 - intensity[..., None]
    rgba[..., 3] = (intensity > 0).astype(np.float32)
    return rgba


def build_views(image):
    brightness_views = [
        (f"Brightness {brightness:.1f}", apply_brightness(image, brightness))
        for brightness in BRIGHTNESS_LEVELS
    ]
    sobel_views = [
        (f"Sobel {brightness:.1f}", sobel_edge(bright_image))
        for brightness, (_, bright_image) in zip(BRIGHTNESS_LEVELS, brightness_views)
    ]

    return brightness_views, sobel_views


def visualize_random_mnist_brightness(
    root="./dataset",
    index=None,
    seed=None,
    save_path=None,
    dpi=300,
    show=False,
    transparent=True,
):
    images, labels = load_mnist_test(root)

    if index is None:
        rng = random.Random(seed)
        index = rng.randrange(len(images))
    elif not 0 <= index < len(images):
        raise ValueError(f"index must be in [0, {len(images) - 1}], got {index}.")

    image = images[index]
    label = int(labels[index])
    brightness_views, sobel_views = build_views(image)

    fig, axes = plt.subplots(2, 4, figsize=(8, 5.0), dpi=dpi)
    if transparent:
        fig.patch.set_alpha(0)

    row_settings = (
        (brightness_views, 255.0),
        (sobel_views, SOBEL_DISPLAY_VMAX),
    )
    for row_axes, (views, vmax) in zip(axes, row_settings):
        for ax, (name, view) in zip(row_axes, views):
            if transparent:
                ax.set_facecolor("none")
                ax.imshow(to_transparent_rgba(view, vmax), interpolation="nearest")
            else:
                ax.imshow(view, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(name, fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        f"MNIST brightness and Sobel after brightness  |  label: {label}  |  index: {index}",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.09, wspace=0.12, hspace=0.35)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            bbox_inches="tight",
            edgecolor="none",
            facecolor="none" if transparent else "white",
            pad_inches=0.02,
            transparent=transparent,
            dpi=dpi,
        )

    if show:
        plt.show()

    return fig, axes, index, label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample one MNIST test image and visualize normal brightness levels.",
    )
    parser.add_argument("--root", type=str, default="./dataset", help="MNIST data root.")
    parser.add_argument("--index", type=int, default=7, help="Use a fixed MNIST test index.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--save-path", type=str, default="./pictures/mnist_brightness_sobel_compare.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Open the figure window after saving.")
    parser.add_argument("--opaque", action="store_true", help="Save with an opaque white figure background.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_random_mnist_brightness(
        root=args.root,
        index=args.index,
        seed=args.seed,
        save_path=args.save_path,
        dpi=args.dpi,
        show=args.show,
        transparent=not args.opaque,
    )
