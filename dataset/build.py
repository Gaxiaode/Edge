import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as T

from .transform import BrightnessTransform, EdgeTransform, RaplaceTransform

def build_train_transforms(args):
    transform = T.Compose([
        BrightnessTransform(brightness=args.brightness, return_type="pil"),
        EdgeTransform(use=args.use_edge, method=args.edge_method),
        T.ToTensor(),
    ])
    return transform


def build_test_transforms(args):
    transform = T.Compose([
        BrightnessTransform(brightness=args.brightness, return_type="pil"),
        RaplaceTransform(args),
        EdgeTransform(use=args.use_edge, method=args.edge_method),
        T.ToTensor(),
    ])
    return transform




def build_dataloader(args, logger):
    """Build MNIST train, validation, and test dataloaders.

    ``args.val_source`` controls the validation source:
    - ``val``: split ``args.val_ratio`` of the official training set.
    - ``test``: use the official test set directly.
    """
    if args.val_source not in {"val", "test"}:
        raise ValueError(f"Unsupported val_source '{args.val_source}'. Expected 'val' or 'test'.")

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available(),}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    if args.use_edge:
        logger.info(f'使用边缘！')
    else:
        logger.info(f'普通模式，不使用边缘')

    if args.training:
        print(f'\n进行训练\n')
        transform = build_train_transforms(args)

        train_dataset = datasets.MNIST(root=args.root_dir, train=True, transform=transform, download=True,)
        test_dataset = datasets.MNIST(root=args.root_dir, train=False, transform=transform, download=True, )
        num_classes = len(train_dataset.classes)

        if args.val_source == "val":
            if not 0 < args.val_ratio < 1:
                raise ValueError(f"val_ratio must be between 0 and 1, but got {args.val_ratio}.")

            val_size = round(len(train_dataset) * args.val_ratio)
            if val_size == 0 or val_size == len(train_dataset):
                raise ValueError("val_ratio produces an empty training or validation dataset.")

            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed),)
        else:
            val_dataset = test_dataset

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs,)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,  **loader_kwargs,)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs,)

        return train_loader, val_loader, test_loader, num_classes
    
    else:
        print(f'\n进行推理\n')
        transform = build_test_transforms(args)

        test_dataset = datasets.MNIST(root=args.root_dir, train=False, transform=transform, download=True, )
        num_classes = len(test_dataset.classes)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs,)

        return test_loader, num_classes
        




