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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['MaskRCNN']


@register
class MaskRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 mask_head,
                 bbox_post_process,
                 mask_post_process,
                 neck=None):
        """
        backbone (nn.Layer): backbone instance.
        rpn_head (nn.Layer): generates proposals using backbone features.
        bbox_head (nn.Layer): a head that performs per-region computation.
        """
        super(MaskRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        # hack
        if mask_head:
            self.mask_head.set_bbox_head(bbox_head)
        self.bbox_post_process = bbox_post_process
        self.mask_post_process = mask_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)

        out_shape = neck and out_shape or head.get_head().out_dim()
        kwargs = {'input_shape': out_shape}
        mask_head = create(cfg['mask_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
            "mask_head": mask_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)

        import numpy as np
        for v in body_feats:
            print('backbone ', v.shape, np.sum(np.abs(v.numpy())))

        if self.neck is not None:
            body_feats = self.neck(body_feats)

            for v in body_feats:
                print('fpn ', v.shape, np.sum(np.abs(v.numpy())))

        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)

            print('rois ', rois.shape, np.sum(np.abs(rois.numpy())))

            bbox_loss = self.bbox_head(body_feats, rois, rois_num, self.inputs)
            rois, rois_num = self.bbox_head.get_assigned_rois()
            mask_loss = self.mask_head(body_feats, rois, rois_num, self.inputs)
            return rpn_loss, bbox_loss, mask_loss
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds = self.bbox_head(body_feats, rois, rois_num, None)
            print('self.mask_head=in=rois ', rois.shape,
                  np.sum(np.abs(rois.numpy())))
            # TODO, more clear returned value
            bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                    self.inputs['im_shape'],
                                                    self.inputs['scale_factor'])
            mask_out = self.mask_head(body_feats, bbox, bbox_num, self.inputs)
            return bbox, bbox_num, mask_out

    def get_loss(self, ):
        bbox_loss, mask_loss, rpn_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        loss.update(mask_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        return dict(zip(['bbox', 'bbox_num', 'mask'], self._forward()))
