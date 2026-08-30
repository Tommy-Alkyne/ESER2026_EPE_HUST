"""3D 超参数入口。

参数定义全部复用 `args.py`，这里只把 `--ndim` 的默认值改成 3，避免两个 parser
逐条漂移（旧版是两份几乎完全重复、又各自缺了对方参数的文件）。

与 `args.py` 一样，`args` / `device` 是惰性求值的，import 本模块不会解析
命令行、不会创建目录、不会写文件。
"""

import torch

from args import (build_parser, parse_args, postprocess,  # noqa: F401
                  prepare_run_dir)


def __getattr__(name):
    if name == 'args':
        value = parse_args(default_ndim=3)
    elif name == 'device':
        value = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
