from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import re


def parse_cfg(cfgfile):
    """Takes a configuration file

    Arguments:
        cfgfile {[type]} -- [description]

    Returns a list of blocks. Each blocks describes a block in the neural network
    to be built. Block is represented as a dictionary in the list
    """

    block = {}
    blocks = []
    with open(cfgfile, "r") as f:
        # get rid of comments and \n
        lines = (line.strip() for line in f if not line.startswith("#"))
        # get rid of whitespace
        lines = (re.sub("\s+", "", line) for line in lines if len(line) > 0)

        for line in lines:
            if line.startswith("["):
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1]
            else:
                key, value = line.split("=")
                block[key] = value
        blocks.append(block)

        return blocks


def create_modules(blocks):
    net_info = blocks[0]
    print(net_info)
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []


if __name__ == "__main__":
    blocks = parse_cfg("./cfg/yolov3.cfg")
    create_modules(blocks)
