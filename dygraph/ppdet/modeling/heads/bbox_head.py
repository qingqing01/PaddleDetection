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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from ppdet.modeling import ops

from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec


@register
class TwoFCHead(nn.Layer):
    def __init__(self, in_dim=256, mlp_dim=1024, resolution=7):
        super(TwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        fan = in_dim * resolution * resolution
        fc6_name = 'fc6_0'
        fc7_name = 'fc7_0'
        lr_factor = 1.
        self.fc6 = self.add_sublayer(
            fc6_name,
            nn.Linear(
                in_dim * resolution * resolution,
                mlp_dim,
                weight_attr=paddle.ParamAttr(
                    learning_rate=lr_factor,
                    initializer=XavierUniform(fan_out=fan)),
                bias_attr=paddle.ParamAttr(
                    learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))
        self.fc7 = self.add_sublayer(
            fc7_name,
            nn.Linear(
                mlp_dim,
                mlp_dim,
                weight_attr=paddle.ParamAttr(
                    learning_rate=lr_factor, initializer=XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_dim': s.channels}

    def out_shape(self):
        return [ShapeSpec(channels=self.mlp_dim, )]

    def forward(self, rois_feat):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6(rois_feat)
        fc6 = F.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = F.relu(fc7)
        return fc7


@register
class BBoxHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner']
    """
    head (nn.Layer):
    in_channel (int):
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=81):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.delta_dim = num_classes

        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)

        self.head = head
        self.bbox_assigner = bbox_assigner
        self.with_pool = with_pool
        self.bbox_feat = None

        score_name = 'bbox_score_0'
        delta_name = 'bbox_delta_0'
        lr_factor = 1.
        self.bbox_score = self.add_sublayer(
            score_name,
            nn.Linear(
                in_channel,
                1 * self.num_classes,
                weight_attr=paddle.ParamAttr(
                    learning_rate=lr_factor,
                    initializer=Normal(
                        mean=0.0, std=0.01)),
                bias_attr=paddle.ParamAttr(
                    learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))

        self.bbox_delta = self.add_sublayer(
            delta_name,
            nn.Linear(
                in_channel,
                4 * self.num_classes,
                weight_attr=paddle.ParamAttr(
                    learning_rate=lr_factor,
                    initializer=Normal(
                        mean=0.0, std=0.001)),
                bias_attr=paddle.ParamAttr(
                    learning_rate=2. * lr_factor, regularizer=L2Decay(0.))))
        self.assigned_label = None
        self.assigned_rois = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape()[0].channels
        }

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]):
        rois (Tensor):
        rois_num (Tensor):
        inputs (dict{Tensor}):
        """
        import numpy as np
        if self.training:
            rois, rois_num, _, targets = self.bbox_assigner(rois, rois_num,
                                                            inputs)
            print('rois-t ', rois.shape, np.sum(np.abs(rois.numpy())))
            self.assigned_rois = (rois, rois_num)
            self.assigned_labels = targets[0]

        for v in body_feats:
            print('bbox-head-roi-in ', v.shape, np.sum(np.abs(v.numpy())))

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        print('bbox-head-in ', rois_feat.shape,
              np.sum(np.abs(rois_feat.numpy())))
        self.bbox_feat = self.head(rois_feat)

        #if self.with_pool:
        if len(self.bbox_feat.shape) > 2 and self.bbox_feat.shape[-1] > 1:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = self.bbox_feat
        print('bbox-head-pred-in ', feat.shape, np.sum(np.abs(feat.numpy())))
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        print('bbox-score ', scores.shape, np.sum(np.abs(scores.numpy())))
        print('bbox-deltas ', deltas.shape, np.sum(np.abs(deltas.numpy())))

        if self.training:
            return self.get_loss(scores, deltas, targets)
        else:
            return self.get_prediction(scores, deltas)

    def get_loss(self, score, delta, target):
        """
        score (Tensor):
        delta (Tensor):
        target (list[Tensor]):
        """
        # TODO: better pass args
        score_tgt, bbox_tgt, inside_w, outside_w = target
        # bbox cls  
        labels_int64 = paddle.cast(x=score_tgt, dtype='int64')
        labels_int64.stop_gradient = True
        loss_bbox_cls = F.softmax_with_cross_entropy(
            logits=score, label=labels_int64)
        loss_bbox_cls = paddle.mean(loss_bbox_cls)
        # bbox reg
        loss_bbox_reg = ops.smooth_l1(
            input=delta,
            label=bbox_tgt,
            inside_weight=inside_w,
            outside_weight=outside_w,
            sigma=1.0)
        loss_bbox_reg = paddle.mean(loss_bbox_reg)
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}
        loss_bbox[cls_name] = loss_bbox_cls
        loss_bbox[reg_name] = loss_bbox_reg
        return loss_bbox

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        delta = paddle.reshape(delta, (-1, self.num_classes, 4))
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    @property
    def feat(self, ):
        assert self.bbox_feat
        return self.bbox_feat

    def get_assigned_labels(self, ):
        return self.assigned_labels

    def get_assigned_rois(self, ):
        return self.assigned_rois
