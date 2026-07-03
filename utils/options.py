import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Edge Args")
    ######################## general settings ########################
    parser.add_argument("--name",                   default="edge_canny", help="experiment name to save")
    parser.add_argument("--output_dir",             default="output")
    parser.add_argument("--checkpoint",             default="", help="checkpoint path used with --test")
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--log_period", type=int,   default=100)
    parser.add_argument("--eval_period", type=int,  default=1)
    parser.add_argument("--val_source", type=str,   default="test",  choices=["val", "test"],)
    parser.add_argument("--val_ratio",  type=float, default=0.15,   help="validation ratio when --val_source=val",)
    parser.add_argument("--model",      type=str,   default='CNN',  choices=['CNN'])
    parser.add_argument("--brightness", type=float, default=1.0,    choices=[1.0, 0.6, 0.2])
    parser.add_argument("--use_edge",   type=bool,  default=True,)
    parser.add_argument("--edge_method",type=str,   default='canny', choices=['sobel', 'prewitt', 'canny'])
    parser.add_argument("--light_path", type=str,   default='./dataset/light')
    
    ######################## loss settings ########################
    parser.add_argument("--loss_names",             default='cross_entropy', choices=['cross_entropy', 'focal_loss'],)
    
    ######################## solver ########################
    parser.add_argument("--optimizer",      type=str,   default="Adam", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum",       type=float, default=0.9)
    parser.add_argument("--weight_decay",   type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha",          type=float, default=0.9)
    parser.add_argument("--beta",           type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch",      type=int,   default=50)
    parser.add_argument("--milestones",     type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma",          type=float, default=0.1)
    parser.add_argument("--warmup_factor",  type=float, default=0.1)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--warmup_method",  type=str,   default="linear")
    parser.add_argument("--lrscheduler",    type=str,   default="cosine")
    parser.add_argument("--target_lr",      type=float, default=0)
    parser.add_argument("--power",          type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name",               default="MNIST")
    parser.add_argument("--root_dir",                   default="./dataset")
    parser.add_argument("--batch_size",      type=int,  default=128)
    parser.add_argument("--test_batch_size", type=int,  default=2048)
    parser.add_argument("--num_classes",     type=int,  default=10)
    parser.add_argument("--num_workers",     type=int,  default=4)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
