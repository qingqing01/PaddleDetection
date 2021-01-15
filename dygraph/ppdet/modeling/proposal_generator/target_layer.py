# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import paddle

from ppdet.core.workspace import register, serializable

from .rpn_target_assign import rpn_anchor_target
from .target import generate_proposal_target, generate_mask_target


@register
@serializable
class RPNTargetAssign(object):
    def __init__(self,
                 batch_size_per_im=256,
                 straddle_thresh=0.,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 use_random=False):
        super(RPNTargetAssign, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.straddle_thresh = straddle_thresh
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.use_random = use_random

    def __call__(self, inputs, anchors):
        """
        inputs: ground-truth instances.
        anchor_box (Tensor): [num_anchors, 4], num_anchors are all anchors in all feature maps.
        """
        gt_boxes = inputs['gt_bbox']
        is_crowd = inputs['is_crowd']
        im_info = inputs['im_info']

        anchors = anchors.numpy()
        gt_boxes = gt_boxes.numpy()
        is_crowd = is_crowd.numpy()
        im_info = im_info.numpy()
        loc_ids, score_ids, tgt_labels, tgt_bboxes, bbox_weights = rpn_anchor_target(
            anchors, gt_boxes, is_crowd, im_info, self.straddle_thresh,
            self.batch_size_per_im, self.positive_overlap,
            self.negative_overlap, self.fg_fraction, self.use_random)

        loc_ids = paddle.to_tensor(loc_ids)
        score_ids = paddle.to_tensor(score_ids)
        tgt_labels = paddle.to_tensor(tgt_labels)
        tgt_bboxes = paddle.to_tensor(tgt_bboxes)
        bbox_weights = paddle.to_tensor(bbox_weights)

        loc_ids.stop_gradient = True
        score_ids.stop_gradient = True
        tgt_labels.stop_gradient = True

        #cls_logits = paddle.reshape(x=cls_logits, shape=(-1, ))
        #bbox_pred = paddle.reshape(x=bbox_pred, shape=(-1, 4))
        #pred_cls_logits = paddle.gather(cls_logits, score_indexes)
        #pred_bbox_pred = paddle.gather(bbox_pred, loc_indexes)
        return loc_ids, score_ids, tgt_labels, tgt_bboxes, bbox_weights


@register
class BBoxAssigner(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=[.5, ],
                 bg_thresh_hi=[.5, ],
                 bg_thresh_lo=[0., ],
                 bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                 num_classes=81,
                 use_random=False,
                 is_cls_agnostic=False):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.num_classes = num_classes
        self.use_random = use_random
        self.is_cls_agnostic = is_cls_agnostic

    def __call__(self,
                 rpn_rois,
                 rpn_rois_num,
                 inputs,
                 stage=0,
                 max_overlap=None):
        rpn_rois = rpn_rois.numpy()
        rpn_rois_num = rpn_rois_num.numpy()
        gt_classes = inputs['gt_class'].numpy()
        gt_boxes = inputs['gt_bbox'].numpy()
        is_crowd = inputs['is_crowd'].numpy()
        im_info = inputs['im_info'].numpy()
        max_overlap = max_overlap and max_overlap.numpy()

        reg_weights = [i / (stage + 1) for i in self.bbox_reg_weights]
        is_cascade = True if stage > 0 else False
        num_classes = 2 if is_cascade else self.num_classes
        # rois, tgt_labels, tgt_deltas, inside_weights
        # outside_weights, rois_num, max_overlaps
        outs = generate_proposal_target(
            rpn_rois, rpn_rois_num, gt_classes, is_crowd, gt_boxes, im_info,
            self.batch_size_per_im, self.fg_fraction, self.fg_thresh[stage],
            self.bg_thresh_hi[stage], self.bg_thresh_lo[stage], reg_weights,
            num_classes, self.use_random, self.is_cls_agnostic, is_cascade,
            max_overlap)
        outs = [paddle.to_tensor(v) for v in outs]
        for v in outs:
            v.stop_gradient = True
        return outs[0], outs[-2], outs[-1], outs[1:5]


@register
@serializable
class MaskAssigner(object):
    __shared__ = ['num_classes', 'mask_resolution']

    def __init__(self, num_classes=81, mask_resolution=14):
        super(MaskAssigner, self).__init__()
        self.num_classes = num_classes
        self.mask_resolution = mask_resolution

    def __call__(self, rois, rois_num, labels_int32, inputs):
        rois = rois.numpy()
        rois_num = rois_num.numpy()
        im_info = inputs['im_info'].numpy()
        gt_classes = inputs['gt_class'].numpy()
        is_crowd = inputs['is_crowd'].numpy()
        gt_segms = inputs['gt_poly'].numpy()
        labels_int32 = labels_int32.numpy()

        outs = generate_mask_target(im_info, gt_classes, is_crowd, gt_segms,
                                    rois, rois_num, labels_int32,
                                    self.num_classes, self.mask_resolution)

        outs = [paddle.to_tensor(v) for v in outs]
        for v in outs:
            v.stop_gradient = True
        # rois, rois_num, rois_has_mask_int32, mask_int32
        return outs
