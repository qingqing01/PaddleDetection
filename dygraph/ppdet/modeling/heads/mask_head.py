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
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from ppdet.modeling import ops

from .roi_extractor import RoIAlign


@register
class MaskFeat(nn.Layer):
    def __init__(self, num_convs=0, in_channels=2048, out_channels=256):
        super(MaskFeat, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        fan_conv = out_channels * 3 * 3
        fan_deconv = out_channels * 2 * 2

        mask_conv = nn.Sequential()
        for j in range(self.num_convs):
            conv_name = 'mask_inter_feat_{}'.format(j + 1)
            mask_conv.add_sublayer(
                conv_name,
                nn.Conv2D(
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    weight_attr=paddle.ParamAttr(
                        initializer=KaimingNormal(fan_in=fan_conv)),
                    bias_attr=paddle.ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            mask_conv.add_sublayer(conv_name + 'act', nn.ReLU())
        mask_conv.add_sublayer(
            'conv5_mask',
            nn.Conv2DTranspose(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=2,
                stride=2,
                weight_attr=paddle.ParamAttr(
                    initializer=KaimingNormal(fan_in=fan_deconv)),
                bias_attr=paddle.ParamAttr(
                    learning_rate=2., regularizer=L2Decay(0.))))
        mask_conv.add_sublayer('conv5_mask' + 'act', nn.ReLU())
        self.upsample = mask_conv

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels, }

    def out_channel(self):
        return self.out_channels

    def forward(self, feats):
        return self.upsample(feats)


@register
class MaskHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['mask_assigner']

    def __init__(self,
                 head,
                 roi_extractor=RoIAlign().__dict__,
                 mask_assigner='MaskAssigner',
                 num_classes=81,
                 share_bbox_feat=False):
        super(MaskHead, self).__init__()
        self.num_classes = num_classes

        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.head = head
        self.in_channels = head.out_channel()
        self.mask_assigner = mask_assigner
        self.share_bbox_feat = share_bbox_feat
        self.bbox_head = None

        self.mask_fcn_logits = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            weight_attr=paddle.ParamAttr(initializer=KaimingNormal(
                fan_in=self.num_classes)),
            bias_attr=paddle.ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.0)))

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
        }

    def set_bbox_head(self, bbox_head):
        """
        Model without FPN share feature extraction with bbox_head
        """
        self.bbox_head = bbox_head

    def forward_train(self, body_feats, rois, rois_num, inputs):
        """
        body_feats (list[Tensor]):
        rois (Tensor):
        rois_num (Tensor):
        inputs (dict):
        """
        import numpy as np
        assert self.bbox_head
        labels_int32 = self.bbox_head.get_assigned_labels()
        print('mask-assigner-rois-in ', rois.shape,
              np.sum(np.abs(rois.numpy())))
        print('labels_int32 ', labels_int32.shape,
              np.sum(np.abs(labels_int32.numpy())))
        rois, rois_num, mask_index, mask_tgt = self.mask_assigner(
            rois, rois_num, labels_int32, inputs)

        for v in body_feats:
            print('mask-in ', v.shape, np.sum(np.abs(v.numpy())))

        print('mask-rois ', rois.shape, np.sum(np.abs(rois.numpy())))
        if self.share_bbox_feat and mask_index is not None:
            bbox_feat = self.bbox_head.feat()
            rois_feat = paddle.gather(bbox_feat, mask_index)
        else:
            rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        print('mask_roi_feat ', rois_feat.shape,
              np.sum(np.abs(rois_feat.numpy())))
        mask_feat = self.head(rois_feat)
        print('mask_feat ', mask_feat.shape, np.sum(np.abs(mask_feat.numpy())))
        mask_logits = self.mask_fcn_logits(mask_feat)

        mask_logits = paddle.flatten(mask_logits, start_axis=1, stop_axis=-1)
        mask_label = paddle.cast(x=mask_tgt, dtype='float32')
        mask_label.stop_gradient = True
        print('mask_logits ', mask_logits.shape,
              np.sum(np.abs(mask_logits.numpy())))
        print('mask_label ', mask_label.shape,
              np.sum(np.abs(mask_label.numpy())))
        loss_mask = ops.sigmoid_cross_entropy_with_logits(
            input=mask_logits,
            label=mask_label,
            ignore_index=-1,
            normalize=True)
        loss_mask = paddle.sum(loss_mask)
        return {'loss_mask': loss_mask}

    def forward_test(self, body_feats, rois, rois_num, scale_factor):
        """
        body_feats (list[Tensor]):
        rois (Tensor):
        rois_num (Tensor):
        scale_factor (Tensor):
        """
        bbox, bbox_num = rois, rois_num
        if bbox.shape[0] == 0:
            mask_out = paddle.full([1, 6], -1)
        else:
            # batch size
            scale_factor_list = []
            for idx in range(bbox_num.shape[0]):
                num = bbox_num[idx]
                scale = scale_factor[idx, 0]
                ones = paddle.ones(num)
                scale_expand = ones * scale
                scale_factor_list.append(scale_expand)
            scale_factor_list = paddle.cast(
                paddle.concat(scale_factor_list), 'float32')
            scale_factor_list = paddle.reshape(scale_factor_list, shape=[-1, 1])
            bbox = paddle.multiply(bbox[:, 2:], scale_factor_list)

            rois_feat = self.roi_extractor(body_feats, rois, bbox_num)
            if self.share_bbox_feat:
                rois_feat = self.bbox_head.get_head()(rois_feat)

            mask_feat = self.head(rois_feat)
            mask_logit = self.mask_fcn_logits(mask_feat)
            mask_out = F.sigmoid(mask_logit)
        return mask_out

    def forward(self, body_feats, rois, rois_num, inputs=None):
        if inputs['mode'] == 'train':
            return self.forward_train(body_feats, rois, rois_num, inputs)
        else:
            im_scale = inputs['scale_factor']
            return self.forward_test(body_feats, rois, rois_num, im_scale)
