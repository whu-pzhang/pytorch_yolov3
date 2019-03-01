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
    prev_filters = 3  # depth of feature map
    output_filters = []

    for index, layer in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # Convolutional layer
        if layer["type"] == "convolutional":
            activation = layer["activation"]
            try:
                batch_normalize = int(layer["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(layer["filters"])
            padding = int(layer["pad"])
            kernel_size = int(layer["size"])
            stride = int(layer["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias)
            module.add_module("conv_{}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(index), activn)
        # Upsampling layer
        elif layer["type"] == "upsample":
            stride = int(layer["stride"])
            upsample = nn.Upsample(scale_factor=stride,  mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        # Route layer
        elif layer["type"] == "route":
            layer["layer"] = layer["layer"].split(",")
            start = int(layer["layer"][0])
            try:
                end = int(layer["layer"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)
            if end < 0:
                # concatenating feature maps
                filters = output_filters[index+start] + \
                    output_filters[index+end]
            else:
                filters = output_filters[index+start]
        # Shotcut corresponds to skip connection
        elif layer["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        # YOLO is the detection layer
        elif layer["type"] == "yolo":
            mask = layer["mask"].split(",")
            mask = [int(val) for val in mask]

            anchors = [int(a) for a in layer["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


if __name__ == "__main__":
    # blocks = parse_cfg("./cfg/yolov3.cfg")
    # create_modules(blocks)
    a = "1,"
    x, y = a.split(",")

    print(y == "")
