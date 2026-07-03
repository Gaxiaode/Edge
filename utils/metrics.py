import os
import random
import logging

import torch
import numpy as np
from prettytable import PrettyTable

from utils.draw import plot_confusion_matrix, plot_class_acc_bar

def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def _extract_logits(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("logits", "output", "predictions", "preds"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]

    if isinstance(output, (tuple, list)):
        for value in reversed(output):
            if torch.is_tensor(value) and value.ndim >= 2:
                return value

    raise TypeError(
        "Model output must be logits, contain a logits tensor, or be a tuple "
        "such as (loss, logits)."
    )


def classification_metrics(loader, model):
    """Evaluate a classification model on one loader and return metrics in percent."""
    model.eval()
    device = next(model.parameters()).device
    confusion_matrix = None

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1).long()

            try:
                output = model(images, labels)
            except TypeError as error:
                try:
                    output = model(images)
                except TypeError:
                    raise error

            logits = _extract_logits(output)

            if logits.ndim != 2:
                raise ValueError(
                    f"Expected logits with shape [batch_size, num_classes], "
                    f"but got {tuple(logits.shape)}."
                )

            if logits.shape[0] != labels.shape[0]:
                raise ValueError("The number of logits and labels must be the same.")

            num_classes = logits.shape[1]

            if confusion_matrix is None:
                confusion_matrix = torch.zeros(
                    (num_classes, num_classes),
                    dtype=torch.long
                )
            elif confusion_matrix.shape[0] != num_classes:
                raise ValueError("The number of model output classes changed between batches.")

            predictions = logits.argmax(dim=1)

            if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
                raise ValueError("Labels must be in the range [0, num_classes).")

            indices = labels.cpu() * num_classes + predictions.cpu()

            confusion_matrix += torch.bincount(
                indices,
                minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    if confusion_matrix is None:
        raise ValueError("Cannot evaluate an empty loader.")

    # 保存原始整数混淆矩阵
    confusion_matrix_counts = confusion_matrix.clone()

    confusion_matrix = confusion_matrix.float()

    true_positive = confusion_matrix.diag()
    support = confusion_matrix.sum(dim=1)       # 每个真实类别的样本数
    predicted = confusion_matrix.sum(dim=0)     # 每个预测类别的样本数
    total = support.sum()

    false_positive = predicted - true_positive
    false_negative = support - true_positive
    true_negative = total - true_positive - false_positive - false_negative

    per_class_precision = true_positive / predicted.clamp_min(1)
    per_class_recall = true_positive / support.clamp_min(1)

    per_class_specificity = true_negative / (
        true_negative + false_positive
    ).clamp_min(1)

    per_class_f1 = (
        2 * per_class_precision * per_class_recall
        / (per_class_precision + per_class_recall).clamp_min(1e-12)
    )

    # 每个数字类别的分类精度
    # 这里的 class acc = 该类别预测正确数 / 该类别真实总数
    per_class_acc = true_positive / support.clamp_min(1)

    # 适合保存为 csv 的数据结构
    class_acc_csv = []
    for class_id in range(num_classes):
        class_acc_csv.append({
            "class": class_id,
            "correct": int(confusion_matrix_counts[class_id, class_id].item()),
            "total": int(confusion_matrix_counts[class_id].sum().item()),
            "acc": float(per_class_acc[class_id].item() * 100),
        })

    return {
        "acc": (true_positive.sum() / total.clamp_min(1) * 100).item(),
        "recall": (per_class_recall.mean() * 100).item(),
        "precision": (per_class_precision.mean() * 100).item(),
        "specificity": (per_class_specificity.mean() * 100).item(),
        "f1": (per_class_f1.mean() * 100).item(),

        "confusion_matrix": confusion_matrix_counts.tolist(),
        "class_acc_csv": class_acc_csv,
    }


class Evaluator:
    def __init__(self, loader, args):
        self.loader = loader
        self.args = args
        self.logger = logging.getLogger("Edge.test")

    def eval(self, model):
        metrics = classification_metrics(self.loader, model)

        summary = PrettyTable(["ACC", "Recall", "Precision", "Specificity", "F1"])
        summary.add_row(
            [
                metrics["acc"],
                metrics["recall"],
                metrics["precision"],
                metrics["specificity"],
                metrics["f1"],
            ]
        )

        for field in summary.field_names:
            summary.custom_format[field] = lambda _, value: f"{value:.3f}"

        self.logger.info("\nClassification summary:\n%s", summary)

        if not self.args.training and self.args.save_fig:
            picture_dir = os.path.join(".", "pictures", self.args.name)
            os.makedirs(picture_dir, exist_ok=True)

            # 1. 保存混淆矩阵图片
            confusion_path = os.path.join(picture_dir, "confusion.png")
            plot_confusion_matrix(
                confusion_matrix=metrics["confusion_matrix"],
                save_path=confusion_path,
                class_names=[str(i) for i in range(len(metrics["confusion_matrix"]))],
                normalize=False
            )

            # 2. 从已经存在的 csv 读取每类 acc，然后画 bar 图
            class_acc_bar_path = os.path.join(picture_dir, "class_acc_bar.png")

            plot_class_acc_bar(
                class_acc_data=metrics["class_acc_csv"],
                save_path=class_acc_bar_path
            )

            self.logger.info("Saved confusion matrix to: %s", confusion_path)
            self.logger.info("Saved class accuracy bar chart to: %s", class_acc_bar_path)

        return metrics["acc"]