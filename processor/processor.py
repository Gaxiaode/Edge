import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):
    if args.log_period <= 0 or args.eval_period <= 0:
        raise ValueError("log_period and eval_period must be positive integers.")

    logger = logging.getLogger("Edge.train")
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=args.output_dir)

    best_acc = float("-inf")
    best_epoch = None

    logger.info("Start training")

    for epoch in range(start_epoch, args.num_epoch + 1):
        start_time = time.time()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        model.train()

        for iteration, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, logits = model(images, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            batch_acc = logits.argmax(dim=1).eq(labels).float().mean().item() * 100
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(batch_acc, batch_size)

            if iteration % args.log_period == 0 or iteration == len(train_loader):
                logger.info(
                    "Epoch[%d/%d] Iteration[%d/%d] Loss: %.4f, ACC: %.3f, LR: %.2e",
                    epoch,
                    args.num_epoch,
                    iteration,
                    len(train_loader),
                    loss_meter.avg,
                    acc_meter.avg,
                    optimizer.param_groups[0]["lr"],
                )

        writer.add_scalar("train/loss", loss_meter.avg, epoch)
        writer.add_scalar("train/acc", acc_meter.avg, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - start_time
        logger.info(
            "Epoch %d done. Loss: %.4f, ACC: %.3f, Time: %.2fs",
            epoch,
            loss_meter.avg,
            acc_meter.avg,
            elapsed,
        )

        if epoch % args.eval_period == 0:
            logger.info("Validation results - Epoch: %d", epoch)
            val_acc = evaluator.eval(model)
            writer.add_scalar("val/acc", val_acc, epoch)

            if get_rank() == 0 and val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                checkpointer.save(
                    "best",
                    epoch=epoch,
                    best_acc=best_acc,
                    num_epoch=args.num_epoch,
                )

        scheduler.step()

    writer.close()

    if best_epoch is None:
        logger.info("Training finished without validation.")
    else:
        logger.info("Best validation ACC: %.3f at epoch %d", best_acc, best_epoch)

    return best_acc


def do_inference(model, test_loader, args):
    logger = logging.getLogger("Edge.test")
    logger.info("Start inference")

    evaluator = Evaluator(test_loader, args)
    return evaluator.eval(model)
