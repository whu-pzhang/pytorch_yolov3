from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


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


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """[summary]

    Arguments:
        prediction {[type]} -- [description]
        confidence {[type]} -- [description]
        num_classes {int} -- number of classes

    Keyword Arguments:
        nms_conf {float} -- NMS IoU threshold (default: {0.4})
    """
    # set the values which below a threshold to zero
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    box_corner = prediction.new(prediction.shape)
    # top-left corner
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # bottom-right corner
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False

    for idx in range(batch_size):
        image_pred = prediction[idx]
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_idx = torch.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_idx, :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])

        for cls in img_classes:
            # perform NMS

            # Get the detections with one particalar class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)

            class_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_idx].view(-1, 7)

            # Sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # Remove the zero entries
                non_zero_idx = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1, 7)
