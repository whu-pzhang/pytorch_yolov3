from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import re
from util import *


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
    module_list = nn.ModuleList()
    prev_filters = 3  # depth of feature map
    output_filters = []

    for index, layer in enumerate(blocks[1:]):
        module = nn.Sequential()
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
                             kernel_size, stride, pad, bias=bias)
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
            upsample = nn.Upsample(scale_factor=stride,  mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)
        # Route layer
        elif layer["type"] == "route":
            layer["layers"] = layer["layers"].split(",")
            start = int(layer["layers"][0])
            try:
                end = int(layer["layers"][1])
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


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = module["type"]
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(val) for val in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), dim=1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info["height"])

                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), dim=1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0  # keep track of where we are in the weight array
        for i, module in enumerate(self.module_list):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                try:
                    batch_normalize = int(
                        self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = module[0]

                if batch_normalize:
                    bn = module[1]
                    # Get the number of weights of BatchNorm layer
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(
                        weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    # Reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

        fp.close()


# ==========

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608, 608))
    img = img[:, :, ::-1].transpose((2, 0, 1))
    img = img[None, :, :, :] / 255.0
    img = torch.from_numpy(img).float()
    return img


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("weights/yolov3.weights")
