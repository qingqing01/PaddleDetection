#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import six
import math
import numpy as np
from numba import jit
from .bbox import *


@jit
def rpn_anchor_target(anchors,
                      gt_boxes,
                      is_crowd,
                      im_info,
                      rpn_straddle_thresh,
                      rpn_batch_size_per_im,
                      rpn_positive_overlap,
                      rpn_negative_overlap,
                      rpn_fg_fraction,
                      use_random=True,
                      anchor_reg_weights=[1., 1., 1., 1.]):
    anchor_num = anchors.shape[0]
    batch_size = gt_boxes.shape[0]

    loc_indexes = []
    cls_indexes = []
    tgt_labels = []
    tgt_deltas = []
    anchor_inside_weights = []

    for i in range(batch_size):

        # TODO: move anchor filter into anchor generator 
        im_height = im_info[i][0]
        im_width = im_info[i][1]
        im_scale = im_info[i][2]
        if rpn_straddle_thresh >= 0:
            anchor_inds = np.where((anchors[:, 0] >= -rpn_straddle_thresh) & (
                anchors[:, 1] >= -rpn_straddle_thresh) & (
                    anchors[:, 2] < im_width + rpn_straddle_thresh) & (
                        anchors[:, 3] < im_height + rpn_straddle_thresh))[0]
            anchor = anchors[anchor_inds, :]
        else:
            anchor_inds = np.arange(anchors.shape[0])
            anchor = anchors

        gt_bbox = gt_boxes[i] * im_scale
        is_crowd_slice = is_crowd[i]
        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_bbox = gt_bbox[not_crowd_inds]

        # Step1: match anchor and gt_bbox
        anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels = label_anchor(anchor,
                                                                       gt_bbox)

        # Step2: sample anchor 
        fg_inds, bg_inds, fg_fake_inds, fake_num = sample_anchor(
            anchor_gt_bbox_iou, labels, rpn_positive_overlap,
            rpn_negative_overlap, rpn_batch_size_per_im, rpn_fg_fraction,
            use_random)

        # Step3: make output  
        loc_inds = np.hstack([fg_fake_inds, fg_inds])
        cls_inds = np.hstack([fg_inds, bg_inds])

        sampled_labels = labels[cls_inds]

        sampled_anchors = anchor[loc_inds]
        sampled_gt_boxes = gt_bbox[anchor_gt_bbox_inds[loc_inds]]
        sampled_deltas = bbox2delta(sampled_anchors, sampled_gt_boxes,
                                    anchor_reg_weights)

        anchor_inside_weight = np.zeros((len(loc_inds), 4), dtype=np.float32)
        anchor_inside_weight[fake_num:, :] = 1

        loc_indexes.append(anchor_inds[loc_inds] + i * anchor_num)
        cls_indexes.append(anchor_inds[cls_inds] + i * anchor_num)
        tgt_labels.append(sampled_labels)
        tgt_deltas.append(sampled_deltas)
        anchor_inside_weights.append(anchor_inside_weight)

    loc_indexes = np.concatenate(loc_indexes)
    cls_indexes = np.concatenate(cls_indexes)
    tgt_labels = np.concatenate(tgt_labels).astype('float32')
    tgt_deltas = np.vstack(tgt_deltas).astype('float32')
    anchor_inside_weights = np.vstack(anchor_inside_weights)

    return loc_indexes, cls_indexes, tgt_labels, tgt_deltas, anchor_inside_weights


@jit
def label_anchor(anchors, gt_boxes):
    iou = bbox_overlaps(anchors, gt_boxes)
    # every gt's anchor's index
    gt_bbox_anchor_inds = iou.argmax(axis=0)
    gt_bbox_anchor_iou = iou[gt_bbox_anchor_inds, np.arange(iou.shape[1])]
    gt_bbox_anchor_iou_inds = np.where(iou == gt_bbox_anchor_iou)[0]

    # every anchor's gt bbox's index 
    anchor_gt_bbox_inds = iou.argmax(axis=1)
    anchor_gt_bbox_iou = iou[np.arange(iou.shape[0]), anchor_gt_bbox_inds]

    labels = np.ones((iou.shape[0], ), dtype=np.int32) * -1
    labels[gt_bbox_anchor_iou_inds] = 1

    return anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels


@jit
def sample_anchor(anchor_gt_bbox_iou,
                  labels,
                  rpn_positive_overlap,
                  rpn_negative_overlap,
                  rpn_batch_size_per_im,
                  rpn_fg_fraction,
                  use_random=True):

    labels[anchor_gt_bbox_iou >= rpn_positive_overlap] = 1
    num_fg = int(rpn_fg_fraction * rpn_batch_size_per_im)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg and use_random:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    else:
        disable_inds = fg_inds[num_fg:]
    labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    num_bg = rpn_batch_size_per_im - np.sum(labels == 1)
    bg_inds = np.where(anchor_gt_bbox_iou < rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg and use_random:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
    else:
        enable_inds = bg_inds[:num_bg]

    fg_fake_inds = np.array([], np.int32)
    fg_value = np.array([fg_inds[0]], np.int32)
    fake_num = 0
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    labels[enable_inds] = 0

    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]

    return fg_inds, bg_inds, fg_fake_inds, fake_num
