import numpy as np
import torch
import torch.nn as nn

class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


def loss(conf_pred, class_pred, bboxs_pred, iou_pred, label, num_classes):
    '''
        total loss
    '''
    # print("conf_pred", conf_pred)
    # print("class_pred", class_pred)
    # print("bbox_pred", bbox_pred)
    # print("label", label)

    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    offset_xy_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    offset_wh_loss_function = nn.MSELoss(reduction='none')
    iou_loss_function = nn.SmoothL1Loss(reduction='none')

    conf_pred = conf_pred[:, :, 0]
    class_pred = class_pred.permute(0, 2, 1)
    offset_xy_pred = bboxs_pred[:, :, :2]
    offset_wh_pred = bboxs_pred[:, :, 2:]
    iou_pred = iou_pred[:, :, 0]
    
    # ground truth
    conf_gt = label[:, :, 0].float()
    obj_gt = label[:, :, 1].float()
    class_gt = label[:, :, 2].long()
    offset_xy_gt = label[:, :, 3:5].float()
    offset_wh_gt = label[:, :, 5:7].float()
    gt_box_scale_weight = label[:, :, 7]
    iou_gt = (gt_box_scale_weight > 0.).float()
    mask_gt = (gt_box_scale_weight > 0.).float()

    batch_size = conf_pred.size(0)

    # confidence loss
    conf_loss = conf_loss_function(conf_pred, conf_gt, obj_gt)
    
    # class loss
    class_loss = torch.sum(cls_loss_function(class_pred, class_gt) * mask_gt) / batch_size
    
    # bounding box loss

    offset_xy_loss = torch.sum(torch.sum(offset_xy_loss_function(offset_xy_pred, offset_xy_gt), dim=-1) * gt_box_scale_weight * mask_gt) / batch_size
    offset_wh_loss = torch.sum(torch.sum(offset_wh_loss_function(offset_wh_pred, offset_wh_gt), dim=-1) * gt_box_scale_weight * mask_gt) / batch_size
    bbox_loss = offset_xy_loss + offset_wh_loss

    iou_loss = torch.sum(iou_loss_function(iou_pred, iou_gt) * mask_gt) / batch_size

    return conf_loss, class_loss, bbox_loss, iou_loss


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i + 1e-14)