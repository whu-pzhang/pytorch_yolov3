from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from utils.utils import parse_cfg, select_device


def create_modules(blocks):
    """
    Constructs module list of layer blocks from module configuration in blocks

    Arguments:
        blocks {list} -- list of blocks

    Returns:
        tuple -- [description]
    """

    net_info = blocks.pop(0)
    img_size = int(net_info["height"])  # input image size
    output_filters = [int(net_info["channels"])]
    module_list = nn.ModuleList()

    for index, layer in enumerate(blocks):
        module = nn.Sequential()

        # Convolutional layer
        if layer["type"] == "convolutional":
            activation = layer["activation"]
            try:
                batch_normalize = int(layer["batch_normalize"])
            except:
                batch_normalize = 0
            filters = int(layer["filters"])
            kernel_size = int(layer["size"])
            pad = (kernel_size - 1) // 2 if int(layer["pad"]) else 0
            stride = int(layer["stride"])

            # add the convolutional layer
            conv = nn.Conv2d(in_channels=output_filters[-1],
                             out_channels=filters,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=pad,
                             bias=not batch_normalize)
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
            upsample = Upsample(scale_factor=stride,  mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # Route layer
        elif layer["type"] == "route":
            layers = [int(x) for x in layer["layers"].split(",")]
            filters = sum([output_filters[i + 1 if i > 0 else i]
                           for i in layers])
            module.add_module("route_{}".format(index), EmptyLayer())

        # Shotcut corresponds to skip connection
        elif layer["type"] == "shortcut":
            filters = output_filters[int(layer["from"])]
            module.add_module("shortcut_{}".format(index), EmptyLayer())

        # YOLO is the detection layer
        elif layer["type"] == "yolo":
            mask = [int(val) for val in layer["mask"].split(",")]
            # Extract anchors
            anchors = [float(a) for a in layer["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            num_classes = int(layer["classes"])
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            module.add_module("yolo_{}".format(index), yolo_layer)

        # Register module list and number of output filters
        module_list.append(module)
        output_filters.append(filters)

    return net_info, module_list


class Upsample(nn.Module):
    """
    Custom Upsample layer (nn.Upsample give deprecated warning message)
    """

    def __init__(self, scale_factor=1, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.ignore_thres = 0.5
        self.img_size = img_size

    def forward(self, x, targets=None):
        batch_size, grid_size = x.size(0), x.size(2)
        stride = self.img_size // grid_size

        # (bs, depth, 13, 13) --> (bs, 3, 13, 13, 85) # (bs, num_anchors, grid, grid, bbox_attrs)
        prediction = x.view(batch_size, self.num_anchors,
                            self.bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # x, y coordinates
        y = torch.sigmoid(prediction[..., 1])
        w, h = prediction[..., 2], prediction[..., 3]  # width, height
        obj_conf = torch.sigmoid(prediction[..., 4])  # object confidence
        cls_prob = torch.sigmoid(prediction[..., 5:])   # class probility

        # Calculate offsets for each grid
        grid_x = torch.arange(grid_size).repeat(
            grid_size, 1).view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
        grid_y = torch.arange(grid_size).repeat(
            grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)

        scaled_anchors = torch.FloatTensor([(aw / stride, ah / stride)
                                            for aw, ah in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(prediction[..., :4].size())
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:
            pass
        else:
            # (bs, num_anchors*grid_grid, bbox_attrs)
            output = torch.cat(
                (
                    pred_boxes.view(batch_size, -1, 4) * stride,
                    obj_conf.view(batch_size, -1, 1),
                    cls_prob.view(batch_size, -1, self.num_classes)
                ),
                -1
            )

        return output


class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_path)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, targets=None):
        is_training = targets is not None
        img_size = x.shape[-1]
        output = []
        layer_outputs = []

        for module_def, module in zip(self.blocks, self.module_list):
            module_type = module_def["type"]
            if module_type in ["convolutional", "upsample"]:
                x = module(x)

            elif module_type == "route":
                layers = [int(val) for val in module_def["layers"].split(",")]

                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layers], dim=1)

            elif module_type == "shortcut":
                from_ = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[from_]

            elif module_type == "yolo":
                if is_training:
                    pass
                else:  # inference
                    x = module(x)
                output.append(x)

            layer_outputs.append(x)

        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weight_path):
        """
        Parse and load the weight

        Arguments:
            weight_path {str} -- 
        """

        fp = open(weight_path, "rb")

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
        for module_def, module in zip(self.blocks, self.module_list):
            module_type = module_def["type"]
            if module_type == "convolutional":
                try:
                    batch_normalize = int(module_def["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = module[0]
                if batch_normalize:
                    bn = module[1]
                    # Get the number of weights of BatchNorm layer
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases]).view_as(bn.bias)
                    bn.bias.data.copy_(bn_biases)
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases]).view_as(bn.weight)
                    bn.weight.data.copy_(bn_weights)
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases]).view_as(bn.running_mean)
                    bn.running_mean.data.copy_(bn_running_mean)
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases]).view_as(bn.running_var)
                    bn.running_var.data.copy_(bn_running_var)
                    ptr += num_bn_biases
                else:
                    # load conv bias
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(
                        weights[ptr:ptr + num_biases]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_biases)
                    ptr += num_biases

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(
                    weights[ptr: ptr + num_weights]).view_as(conv.weight)
                conv.weight.data.copy_(conv_weights)
                ptr += num_weights

        fp.close()


# ==========

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608, 608))
    img = img[:, :, ::-1].transpose((2, 0, 1))
    img = img[None, :, :, :] / 255.0
    img = torch.from_numpy(img).float()
    return img


def test(cfg, weights):

    device = select_device()
    print("Loading network...")
    model = Darknet(cfg)
    model.load_weights(weights)
    print("Network successfully loaded")

    model.to(device).eval()


if __name__ == "__main__":

    test("cfg/yolov3.cfg", "weights/yolov3.weights")
