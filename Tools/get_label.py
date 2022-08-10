import numpy as np
import torch
from Config.train_config import *


ignore_thresh = IGNORE_THRESH

def set_anchors(anchors_size):
    ''' generate the anchor box: [0, 0, h, w]
        input anchors_size:list -> [[h1, w1],
                                   [h2, w2],
                                    ...
                                   [hn, wn]]
        output anchor_bboxes:ndarray -> [[0, 0, h1, w1],
                                        [0, 0, h2, w2],
                                         ...
                                        [0, 0, hn, wn]]
    '''
    num_anchors = len(anchors_size)
    anchor_bboxes = np.zeros([num_anchors, 4])

    for index, size in enumerate(anchors_size):
        anchor_w, anchor_h = size
        anchor_bboxes[index] = np.array([0, 0, anchor_w, anchor_h])
    return anchor_bboxes

def compute_IoU(anchor_bboxes, gt_bboxes):
    """compute IoU between anchor and ground truth
    Input: 
        anchor_bboxes: [K, 4]
            gt_box: [1, 4]
    Output: 
                iou : [K,]
    """

    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_bboxes), 4])
    # 计算先验框的左上角点坐标和右下角点坐标
    ab_x1y1_x2y2[:, 0] = anchor_bboxes[:, 0] - anchor_bboxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_bboxes[:, 1] - anchor_bboxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_bboxes[:, 0] + anchor_bboxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_bboxes[:, 1] + anchor_bboxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_bboxes[:, 2], anchor_bboxes[:, 3]
    
    # gt_box : 
    # 我们将真实框扩展成[K, 4], 便于计算IoU. 
    gt_box_expand = np.repeat(gt_bboxes, len(anchor_bboxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_bboxes), 4])
    # 计算真实框的左上角点坐标和右下角点坐标
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # 计算IoU
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU

def generate_offset_xywh(gt_label, w, h, s, anchors_size):
    '''
        generate the label for training from gt_label 
        input:
            gt_label[]

            orginal size
            stride
        return:
            label for training
    '''
    xmin, ymin, xmax, ymax = gt_label[:-1]

    # get center coordinate and unnormallized w h of bbox
    gt_center_x = (xmax + xmin) / 2 * w
    gt_center_y = (ymax + ymin) / 2 * h
    gt_bbox_w = (xmax - xmin) * w
    gt_bbox_h = (ymax - ymin) * h

    if gt_bbox_w < 1e-4 or gt_bbox_h < 1e-4:
        return False
    # get the gridcell coords according to bbox's center-coord
    gt_center_x_s = gt_center_x / s
    gt_center_y_s = gt_center_y / s
    gt_bbox_w_s = gt_bbox_w / s
    gt_bbox_h_s = gt_bbox_h / s

    gt_gridcell_x = int(gt_center_x_s)
    gt_gridcell_y = int(gt_center_y_s)

    # set all the bboxes center coords to (0, 0)
    anchor_bboxes = set_anchors(anchors_size)
    gt_bboxes = np.array([[0, 0, gt_bbox_w_s, gt_bbox_h_s]])

    # compute the IoU between ground truth and anchor bbox
    IoU = compute_IoU(anchor_bboxes, gt_bboxes)

    # keep the box whose IoU greater than  ignore threshold
    iou_mask = (IoU > ignore_thresh)

    results = []
    # All anchor's IoU are smaller than ignore thersh
    if iou_mask.sum() == 0:
        index = np.argmax(IoU)
        p_w, p_h = anchors_size[index]
        offset_x = gt_center_x_s - gt_gridcell_x
        offset_y = gt_center_y_s - gt_gridcell_y
        offset_w = np.log(gt_bbox_w_s / p_w)
        offset_h = np.log(gt_bbox_h_s / p_h)
        weight = 2.0 - (gt_bbox_w / w) * (gt_bbox_h / h)

        results.append([index, gt_gridcell_x, gt_gridcell_y, offset_x, offset_y, offset_w, offset_h, weight, xmin, ymin, xmax, ymax])
        return results
    # At least one anchor box exists with a IoU greater than ignore thresh
    else:
        max_index = np.argmax(iou_mask)
        for index, iou_m in enumerate(iou_mask):
            if iou_m == max_index:
                p_w, p_h = anchors_size[index]
                offset_x = gt_center_x_s - gt_gridcell_x
                offset_y = gt_center_y_s - gt_gridcell_y
                offset_w = np.log(gt_bbox_w / p_w)
                offset_h = np.log(gt_bbox_h / p_h)
                weight = 2.0 - (gt_bbox_w / w) * (gt_bbox_h / h)

                results.append([index, gt_gridcell_x, gt_gridcell_y, offset_x, offset_y, offset_w, offset_h, weight, xmin, ymin, xmax, ymax])
            else:
                results.append([index, gt_gridcell_x, gt_gridcell_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])
        return results

def get_groundtruth(input_size, stride, labels_list, anchors_size):
    '''
        return tensor for training
        label_lists:  [img1_label, img2_label, img3_label, ...]
        img1_label:   [obj1, obj2, obj3, ...]
        obj:          [class_index, x, y, h, w]
    '''
    batch_size = len(labels_list)
    w = input_size
    h = input_size
    ws = w // stride
    hs = h // stride
    num_anchors = len(anchors_size)

    gt_tensor = np.zeros([batch_size, hs, ws, num_anchors, 1+1+4+1+4])

    for batch_index in range(batch_size):
        for gt_label in labels_list[batch_index]:
            
            gt_class = int(gt_label[-1])
            results = generate_offset_xywh(gt_label, w, h, stride, anchors_size)
            
            if results:
                for result in results:
                    index, gt_gridcell_x, gt_gridcell_y, gt_offset_x, gt_offset_y, gt_offset_w, gt_offset_h, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0:
                        if gt_gridcell_x < gt_tensor.shape[2] and gt_gridcell_y < gt_tensor.shape[1]:
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 0] = 1.0
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 1] = gt_class
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 2:6] = np.array([gt_offset_x, gt_offset_y, gt_offset_w, gt_offset_h]) 
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 6] = weight
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 7:] = np.array([xmin, ymin, xmax, ymax])
                    else:
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 0] = -1.0
                            gt_tensor[batch_index, gt_gridcell_y, gt_gridcell_x, index, 6] = -1.0

    gt_tensor = gt_tensor.reshape(batch_size, hs * ws * num_anchors, 1+1+4+1+4)

    return torch.from_numpy(gt_tensor)
    