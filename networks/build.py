import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.cnn import CNN


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        probability = torch.exp(-cross_entropy)
        return ((1 - probability) ** self.gamma * cross_entropy).mean()


class ModelEngine(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        # loss
        if args.loss_names == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_names == 'focal_loss':
            self.criterion = FocalLoss()
        else:
            raise ValueError(f"Unsupported loss '{args.loss_names}'")

    def forward(self, img, label):
        # 输入图像 返回损失值
        self.logits = self.model(img)
        self.loss = self.criterion(self.logits, label)
        
        return self.loss, self.logits

def build_model(args):
    if args.model == 'CNN':
        model = CNN(args.num_classes)
    else:
        raise ValueError(f"Unsupported model '{args.model}'")

    return ModelEngine(model, args)
