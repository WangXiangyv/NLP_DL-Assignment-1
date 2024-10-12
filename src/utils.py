import os
import logging
import torch
from torch import nn
from rich.logging import RichHandler
from rich.console import Console
from transformers import HfArgumentParser
import sys

logger = logging.getLogger(__name__)


def save_model(model:nn.Module, save_dir:os.PathLike, prefix:str, epoch:int):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{prefix}_epoch_{epoch}"))
    
def set_logger(level:str|int="WARNING"):
    logging.basicConfig(
        level=level,
        format="%(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[RichHandler(level=level, markup=True, rich_tracebacks=True, console=Console())]
    )
    
def parse_arguments(*datacls):
    parser = HfArgumentParser(datacls)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    return args