from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import re


def parse_cfg(cfg_path):
    """
    Parses the yolov3 layer configuration file

    Arguments:
        cfg_path {str} -- cfg file path

    Returns a list of blocks. Each blocks describes a block in the neural network
    to be built. Block is represented as a dictionary in the list
    """

    blocks = []
    with open(cfg_path, "r") as f:
        # get rid of comments and \n
        lines = (line.strip() for line in f if not line.startswith("#"))
        # get rid of whitespaces
        lines = (re.sub("\s+", "", line) for line in lines if len(line) > 0)

        for line in lines:
            if line.startswith("["):  # start of a new block
                blocks.append({})
                blocks[-1]["type"] = line[1:-1]
            else:
                key, value = line.split("=")
                blocks[-1][key] = value
        return blocks


def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device("cpu")
    else:
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda else "cpu")

    print("Using {} {}".format(device.type,
                               torch.cuda.get_device_properties(0) if cuda else ""))
    return device


def non_max_suppression(prediction, conf_thres, nms_thres):

    # Convert (center_x, center_y, w, h) to
    # top-left corner (x1,y1) and bottom-right corner (x2,y2)
    box_corner = prediction.new(prediction.shape)
    # top-left corner
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # bottom-right corner
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    write = False
    output = [None for _ in range(len(prediction))]
    for i, pred in enumerate(prediction):
        class_prob, class_pred = torch.max(pred[:, 5:], 1)

        non_zero_index = torch.nonzero(pred[:, 4]).squeeze()
        pred = pred[non_zero_index]
        class_prob = class_prob[non_zero_index].float().unsqueeze(1)
        class_pred = class_pred[non_zero_index].float().unsqueeze(1)

        # if none are remaining --> process next image
        if pred.shape[0] == 0:
            continue

        seq = (pred[:, :5], class_prob, class_pred)
        detections = torch.cat(seq, 1)

        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            _, conf_sort_index = torch.sort(
                dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppresion
            # num_detections = dc.shape[0]
            # for j in range(num_detections):
            #     try:
            #         # IoU with orther boxes
            #         ious = bbox_iou(dc[j].unsqueeze(0), dc[j+1:])
            #         print(ious.size())
            #     except ValueError:
            #         break
            #     except IndexError:
            #         break
            #     # remove ious > threshold entry
            #     dc = dc[j+1:][ious < nms_thres]
            det_max = []
            while dc.shape[0]:
                det_max.append(dc[:1])
                if len(dc) == 1:
                    break
                ious = bbox_iou(det_max[-1], dc[1:])
                dc = dc[1:][ious < nms_thres]

            # batch_idx = dc.new(len(dc), 1).fill_(i)
            # seq = batch_idx, dc
            # if not write:
            #     output = torch.cat(seq, 1)
            #     write = True
            # else:
            #     out = torch.cat(seq, 1)
            #     output = torch.cat((output, out))

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                if not write:
                    output[i] = det_max
                    write = True
                else:
                    output[i] = torch.cat((output[i], det_max))

    return output


#=============================


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def load_classes(namesfile):
    with open(namesfile, "r") as fp:
        names = [line.strip() for line in fp]
        return names


def letterbox_image(image, inp_dim):
    """Resize image with unchanged aspect ratio

    Arguments:
        img {ndarray} -- [description]
        inp_dim {tuple} -- [description]

    Returns:
        canvas {torch.Tensor}
    """
    shape = image.shape[0:2]  # [height, width]
    w, h = inp_dim
    scale = float(h) / max(shape)  # scale factor = old / new
    new_shape = (round(shape[1] * scale), round(shape[0] * scale))
    dh = (h - new_shape[1]) / 2  # height padding
    dw = (h - new_shape[0]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(resized_image, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img
