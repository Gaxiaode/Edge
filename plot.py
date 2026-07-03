import argparse
from pathlib import Path

from utils.draw import plot_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 ./data/name 下的 CSV 并逐个绘图保存到 ./pictures/name")

    parser.add_argument("--name",           type=str, default='edge', help="如 normal 表示读取 ./data/normal")
    parser.add_argument("--data-root",      type=str, default="./data")
    parser.add_argument("--picture-root",   type=str, default="./pictures")
    parser.add_argument("--recursive", action="store_true", help="递归查找 CSV")

    parser.add_argument("--x-axis",         type=str, default="step", choices=["step", "timestamp"])
    parser.add_argument("--encoding",       type=str, default="utf-8")

    parser.add_argument("--dpi",            type=int,   default=500)
    parser.add_argument("--figsize",        type=float, default=(8, 5), nargs=2)
    parser.add_argument("--color",          type=str,   default='red')
    parser.add_argument("--linewidth",      type=float, default=1.8)
    parser.add_argument("--marker",         type=str,   default="")

    parser.add_argument("--title",          type=str,   default=None)
    parser.add_argument("--xlabel",         type=str,   default=None)
    parser.add_argument("--ylabel",         type=str,   default=None)
    parser.add_argument("--label",          type=str,   default=None)
    parser.add_argument("--show-label", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--font-family",    type=str,   default="Microsoft YaHei")
    parser.add_argument("--font-size",      type=int,   default=16)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transparent",    type=bool,  default=True)
    parser.add_argument("--save-format",    type=str,   default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"])

    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.name
    save_dir = Path(args.picture_root) / args.name

    if not data_dir.exists(): 
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    csv_files = sorted(data_dir.glob("**/*.csv" if args.recursive else "*.csv"))
    if not csv_files: 
        raise FileNotFoundError(f"没有找到 CSV 文件：{data_dir}")

    for csv_path in csv_files:
        plot_csv(csv_path, save_dir, args.x_axis, args.dpi, tuple(args.figsize),
                 args.color, args.linewidth, args.marker, args.show_label,
                 args.label, args.title, args.xlabel, args.ylabel,
                 args.font_family, args.font_size, args.grid,
                 args.save_format, args.encoding, args.transparent)