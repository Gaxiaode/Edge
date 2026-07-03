import os.path as op
import random
import time

import numpy as np
import torch

from utils.comm import get_rank
from utils.logger import setup_logger
from utils.options import get_args
from utils.metrics import Evaluator
from utils.iotools import save_train_configs
from utils.checkpointer import Checkpointer

from solver import build_optimizer, build_lr_scheduler
from dataset import build_dataloader
from networks import build_model
from processor.processor import do_train

def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    name = args.name

    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{name}_{cur_time}')
    logger = setup_logger('Edge', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using device %s with %d GPU(s)", device, num_gpus)
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    train_loader, val_loader, test_loader, num_classes = build_dataloader(args, logger)
    args.num_classes = num_classes
    model = build_model(args)
    logger.info('Total params: %4.fk' % (sum(p.numel() for p in model.parameters()) / 1000.0))
    model.to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(
        model,
        optimizer,
        scheduler,
        args.output_dir,
        is_master,
        logger=logger,
    )
    evaluator = Evaluator(val_loader, args)
    test_evaluator = Evaluator(test_loader, args)

    if not args.training:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when using --test.")
        checkpointer.load(args.checkpoint)
        logger.info("Test results")
        test_evaluator.eval(model)
    else:
        start_epoch = 1
        do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)

        best_checkpoint = op.join(args.output_dir, "best.pth")
        if op.exists(best_checkpoint):
            checkpointer.load(best_checkpoint)
            logger.info("Test results using the best validation checkpoint")
            test_evaluator.eval(model)
