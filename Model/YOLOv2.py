import numpy as np
import torch
import torch.nn as nn
from Backbone.resnet import resnet50
from Modules.common import CBL, SPP, Reorg_layer
import Tools.loss_function as loss_function


class YOLOv2(nn.Module):
    def __init__(self, device, num_classes=20, is_train=False, input_size=None, conf_thresh=0.01, nms_thresh=0.5, anchors_size=None):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors_size)
        self.stride = 32
        self.device = device
        self.anchors_size = torch.tensor(anchors_size)
        self.grid_cell, self.all_anchor_wh = self.get_grid_matrix(input_size)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.is_train = is_train

        # backbone
        self.backbone = resnet50(pretrained=is_train)
        # channel of output
        # c5 = 2048

        # neck
        self.neck = nn.Sequential(
            # downsample
            CBL(2048, 1024, kernel_size=1),
            CBL(1024, 1024, kernel_size=3, padding=1),
            CBL(1024, 1024, kernel_size=1)
        )

        # passthrough(from paper)
        # multi scale featuremap merging
        self.route_layer = CBL(1024, 128, kernel_size=1)
        # 13*13*1536
        self.reorg = Reorg_layer(stride=2)

        # detection head
        self.head = CBL(1024+128*4, 1024, kernel_size=3, padding=1)

        # prediction head
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

    def get_grid_matrix(self, input_size):
        """
            get matrix for all coordinate of grid center (x,y)
        """
        w, h = input_size, input_size
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        grid_xy_matrix = grid_xy.view(1, hs*ws, 1, 2).to(self.device)
        anchors_wh = self.anchors_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy_matrix, anchors_wh

    def reset_grid_matrix(self, input_size):
        """
            for reset G matrix
        """
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.get_grid_matrix(input_size)

    def decode_offset_xywh(self, offset_xywh_pred):
        '''
            decode offset_xywh to real xywh
        '''

        B, HW, ab_n, _ = offset_xywh_pred.size()

        offset_xy_pred = torch.sigmoid(offset_xywh_pred[:, :, :, :2]) + self.grid_cell

        offset_wh_pred = torch.exp(offset_xywh_pred[:, :, :, 2:]) * self.all_anchor_wh

        xywh_pred = torch.cat([offset_xy_pred, offset_wh_pred], -1).view(B, HW*ab_n, 4) * self.stride

        return xywh_pred 

    def decode_boxes(self, offset_xywh_pred, requires_grad=False):
        """
        Decode the bbox_pred to real bounding box
            input: 
                offset_xywh_pred:[B, H*W, anchor_n, (offset_x, offset_y, offset_w, offset_h)]
            return: 
                bbox_pred:[B, H*W, anchor_n, (xmin, ymin, xmax, xmax)]
        """

        xywh_pred = self.decode_offset_xywh(offset_xywh_pred)

        bbox_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        bbox_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        bbox_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        bbox_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        bbox_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return bbox_pred

    def nms(self, dets, scores):
        """"
            Pure Python NMS baseline.
        """
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        

        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
            bboxes: (HxW, 4), bsize = 1
            scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # filter by threshold 
        # index 59 is out of bounds for axis 0 with size 1
        keep = np.where(scores >= self.conf_thresh)     # return index which meet the criterion of conditional expression
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS for each class
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        # get result
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):

        _, C_4, C_5 = self.backbone(x)

        P_5 = self.neck(C_5)
        # handle the C_4 feature map
        P_4 = self.reorg(self.route_layer(C_4))

        P_5 = torch.cat([P_4, P_5], dim=1)
        # head
        P_5 = self.head(P_5)

        pred = self.pred(P_5)

        B, abC, H, W = pred.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C] 
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)
        
        # confidence prediction
        # [B, H*W*num_anchors, 1]
        _conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # class prediction
        # [B, H*W*num_anchors, num_classes]
        _class_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)

        # bounding box prediction
        # [B, H*W, num_anchor * 4], include x_offset, y_offset, w, h
        _offset_xywh_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

        # train
        if self.is_train:
            _offset_xywh_pred = _offset_xywh_pred.view(B, H*W, self.num_anchors, 4)
            bbox_pred = (self.decode_boxes(offset_xywh_pred=_offset_xywh_pred) / self.input_size).view(-1, 4)
            bbox_gt = targets[:, :, 7:].view(-1, 4)

            # IoU computation
            iou_pred = loss_function.iou_score(bbox_pred, bbox_gt).view(B, -1, 1)
            # set IoU as the learning target of confidence
            with torch.no_grad():
                conf_gt = iou_pred.clone()

            _offset_xywh_pred = _offset_xywh_pred.view(B, H*W*self.num_anchors, 4)
            targets = torch.cat([conf_gt, targets[:, :, :7]], dim=2)

            # print(bbox_pred.shape)
            conf_loss, class_loss, bbox_loss, iou_loss = loss_function.loss(conf_pred=_conf_pred,
                                                                            class_pred=_class_pred,
                                                                            bboxs_pred=_offset_xywh_pred,
                                                                            iou_pred=iou_pred,
                                                                            label=targets,
                                                                            num_classes=self.num_classes)
            return conf_loss, class_loss, bbox_loss, iou_loss
        else:
            # test
            bboxs_pred = bbox_pred.view(B, H*W, self.num_anchors, 4)
            with torch.no_grad():
                # [B, H*W, 1] -> [H*W, 1]
                conf_pred = torch.sigmoid(_conf_pred)[0]
                # [B, H*W, 4] -> [H*W, 4]
                # clamp() for limiting the value of bbox_pred in range 0 to 1
                bboxes = torch.clamp((self.decode_boxes(bboxs_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W, 1] -> [H*W, 1]
                scores = torch.softmax(_class_pred[0, :, :], dim=1) * conf_pred
                # move var to cpu for postprocess 
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # postprocess
                bboxes, scores, class_inds = self.postprocess(bboxes, scores)
                return bboxes, scores, class_inds

