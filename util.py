from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes):
    """Transform a detection feature map into 2D tensor

    Arguments:
        prediction {torch.Tensor} -- feature map of detection layer (N, C, H, W)
        inp_dim {int} -- input dimension
        anchors {list} -- list of anchors
        num_classes {int} -- number of classes
        device {str} -- "cuda" or "cpu"
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # print("inp_dim={}; stride={}; grid_size={}; bbox_attrs={}; H={}".format(
    #     inp_dim, stride, grid_size, bbox_attrs, prediction.size(2)))

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the center_x, center_Y and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_y_offset = torch.cat((x_offset, y_offset), dim=1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and width
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid(prediction[:, :, 5: 5 + num_classes])
    # resize the detection map to the size of the input image
    prediction[:, :, :4] *= stride

    return prediction
