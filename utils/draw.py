import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(csv_path, encoding="utf-8"):
    df = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2], encoding=encoding)
    df.columns = ["timestamp", "step", "value"]
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["step", "value"])


def plot_csv(csv_path, save_dir, x_axis="step", dpi=600, figsize=(8, 5), color=None,
             linewidth=1.8, marker="", show_label=True, label=None, title=None,
             xlabel=None, ylabel=None, font_family="Microsoft YaHei", font_size=12,
             grid=True, save_format="png", encoding="utf-8", transparent=False):

    df = read_csv(csv_path, encoding)
    if df.empty: print(f"[跳过] {csv_path} 无有效数据"); return

    if x_axis == "step":
        x, default_xlabel = df["step"], "Step"
    elif x_axis == "timestamp":
        x, default_xlabel = pd.to_numeric(df["timestamp"], errors="coerce"), "Timestamp"
        if x.isna().all(): x = pd.to_datetime(df["timestamp"], errors="coerce")
        valid = ~pd.isna(x); x, df = x[valid], df[valid]
    else:
        raise ValueError("x_axis 只能是 step 或 timestamp")

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, df["value"], color=color, linewidth=linewidth, marker=marker,
            label=(label or csv_path.stem) if show_label else None)

    ax.set_title(title or csv_path.stem)
    ax.set_xlabel(xlabel or default_xlabel)
    ax.set_ylabel(ylabel or csv_path.stem)

    if grid: ax.grid(True, linestyle="--", alpha=0.4)
    if show_label: ax.legend()

    save_dir.mkdir(parents=True, exist_ok=True)  # pictures/name 不存在就自动创建
    save_path = save_dir / f"{csv_path.stem}.{save_format}"

    fig.tight_layout()
    fig.savefig(save_path, transparent=transparent)
    plt.close(fig)
    print(f"[保存] {save_path}")



def plot_confusion_matrix(confusion_matrix, save_path, class_names=None, normalize=False):
    """
    Plot and save confusion matrix.

    confusion_matrix: list 或 numpy array，形状 [num_classes, num_classes]
                      行表示真实类别，列表示预测类别
    save_path: 图片保存路径，例如 ./pictures/MNIST/confusion.png
    class_names: 类别名称，默认 0,1,2,...
    normalize: 是否按行归一化为百分比
    """
    cm = np.asarray(confusion_matrix)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix, but got shape {cm.shape}.")

    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(
            cm,
            np.clip(row_sum, 1, None),
            out=np.zeros_like(cm, dtype=float),
            where=row_sum != 0
        ) * 100
    else:
        display_cm = cm

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.imshow(display_cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = display_cm.max() / 2 if display_cm.max() > 0 else 0

    for i in range(num_classes):
        for j in range(num_classes):
            if normalize:
                text = f"{display_cm[i, j]:.1f}"
            else:
                text = str(int(display_cm[i, j]))

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if display_cm[i, j] > threshold else "black",
                fontsize=8
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_class_acc_bar(class_acc_data, save_path):
    """
    直接使用 classification_metrics 返回的 metrics["class_acc_csv"] 画柱状图。

    class_acc_data 格式：
    [
        {"class": 0, "correct": 980, "total": 1000, "acc": 98.0},
        {"class": 1, "correct": 990, "total": 1000, "acc": 99.0},
        ...
    ]
    """
    if not class_acc_data:
        raise ValueError("class_acc_data is empty.")

    classes = []
    accs = []

    for item in class_acc_data:
        classes.append(str(item["class"]))
        accs.append(float(item["acc"]))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=600)

    bars = ax.bar(classes, accs, color='blue')

    ax.set_title("Per-class Accuracy")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(99, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.font_size = 12

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)