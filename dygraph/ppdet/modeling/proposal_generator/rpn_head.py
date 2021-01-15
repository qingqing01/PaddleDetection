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
from paddle.nn.initializer import Normal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling import ops

from .anchor_generator import AnchorGenerator
from .target_layer import RPNTargetAssign
from .proposal_generator import ProposalGenerator


class RPNFeat(nn.Layer):
    def __init__(self, feat_in=1024, feat_out=1024):
        super(RPNFeat, self).__init__()
        # rpn feat is shared with each level
        self.rpn_conv = nn.Conv2D(
            in_channels=feat_in,
            out_channels=feat_out,
            kernel_size=3,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(F.relu(self.rpn_conv(feat)))
        return rpn_feats


@register
class RPNHead(nn.Layer):
    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_proposal=ProposalGenerator(12000, 2000).__dict__,
                 test_proposal=ProposalGenerator().__dict__,
                 in_channel=1024):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            print(train_proposal)
            self.train_proposal = ProposalGenerator(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = ProposalGenerator(**test_proposal)

        num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(in_channel, in_channel)
        # rpn head is shared with each level
        # rpn roi classification scores
        self.rpn_rois_score = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

        # rpn roi bbox regression deltas
        self.rpn_rois_delta = nn.Conv2D(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

    @classmethod
    def from_config(cls, cfg, input_shape):
        # FPN share same rpn head
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, feats, inputs):
        rpn_feats = self.rpn_feat(feats)
        scores = []
        deltas = []

        import numpy as np

        for rpn_feat in rpn_feats:
            rrs = self.rpn_rois_score(rpn_feat)
            rrd = self.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)

            print('rpn-f', rrs.shape, np.sum(np.abs(rrs.numpy())))

        anchors = self.anchor_generator(rpn_feats)

        for v, _ in anchors:
            print('anchor ', v.shape, np.sum(np.abs(v.numpy())))

        rois, rois_num = self._gen_proposal(inputs, scores, deltas, anchors)

        print('roi-g ', rois.shape, np.sum(np.abs(rois.numpy())))

        if self.training:
            loss = self.get_loss(scores, deltas, anchors, inputs)
            return rois, rois_num, loss
        else:
            return rois, rois_num, None

    def _gen_proposal(self, inputs, scores, bbox_deltas, anchors):
        """
        inputs (): 
        scores (): 
        bbox_deltas (): 
        anchors (): 
        """
        prop_gen = self.train_proposal if self.training else self.test_proposal
        # TODO: delete im_info 
        try:
            im_shape = inputs['im_info']
        except:
            im_shape = inputs['im_shape']
        rpn_rois_list = []
        rpn_prob_list = []
        rpn_rois_num_list = []
        import numpy as np
        for rpn_score, rpn_delta, (anchor, var) in zip(scores, bbox_deltas,
                                                       anchors):
            rpn_prob = F.sigmoid(rpn_score)
            print('===_gen_proposal====p ', rpn_prob.shape,
                  np.sum(np.abs(rpn_prob.numpy())))
            print('===_gen_proposal====b ', rpn_delta.shape,
                  np.sum(np.abs(rpn_delta.numpy())))
            print('===_gen_proposal====anchor ', anchor.shape,
                  np.sum(np.abs(anchor.numpy())))
            rpn_rois, rpn_rois_prob, rpn_rois_num = prop_gen(
                scores=rpn_prob,
                bbox_deltas=rpn_delta,
                anchors=anchor,
                variances=var,
                im_shape=im_shape)
            print('===_gen_proposal====roi ', rpn_rois.shape,
                  np.sum(np.abs(rpn_rois.numpy())))
            print('===_gen_proposal====rpn_rois_prob ', rpn_rois_prob.shape,
                  np.sum(np.abs(rpn_rois_prob.numpy())))
            rpn_rois_list.append(rpn_rois)
            rpn_prob_list.append(rpn_rois_prob)
            rpn_rois_num_list.append(rpn_rois_num)

        if len(bbox_deltas) == 1:
            rois, rois_num = rpn_rois_list[0], rpn_rois_num_list[0]
        else:
            post_nms_top_n = prop_gen.post_nms_top_n
            start_level = 2
            end_level = start_level + len(bbox_deltas)
            rois, rois_num = ops.collect_fpn_proposals(
                rpn_rois_list,
                rpn_prob_list,
                start_level,
                end_level,
                post_nms_top_n,
                rois_num_per_level=rpn_rois_num_list)
        return rois, rois_num

    def get_loss(self, pred_scores, pred_deltas, anchors, inputs):
        """
        pred_scores (list[Tensor]):
        pred_deltas (list[Tensor]):
        anchors (list[Tensor]):
        inputs (dict): input instance, including im, gt_bbox, gt_score
        """
        anchors = [paddle.reshape(a, shape=(-1, 4)) for a, _ in anchors]
        anchors = paddle.concat(anchors)

        scores = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 1)) for v in pred_scores
        ]
        scores = paddle.concat(scores, axis=1)

        deltas = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 4)) for v in pred_deltas
        ]
        deltas = paddle.concat(deltas, axis=1)

        loc_ids, score_ids, score_tgt, loc_tgt, bbox_w = self.rpn_target_assign(
            inputs, anchors)

        scores = paddle.reshape(x=scores, shape=(-1, ))
        deltas = paddle.reshape(x=deltas, shape=(-1, 4))
        scores = paddle.gather(scores, score_ids)
        deltas = paddle.gather(deltas, loc_ids)

        import numpy as np
        print('rpn-pred-score ', scores.shape, np.sum(np.abs(scores.numpy())))
        print('rpn-score-tgt ', scores.shape, np.sum(np.abs(score_tgt.numpy())))

        # cls loss
        score_tgt = paddle.cast(x=score_tgt, dtype='float32')
        score_tgt.stop_gradient = True
        loss_rpn_cls = ops.sigmoid_cross_entropy_with_logits(
            input=scores, label=score_tgt)
        loss_rpn_cls = paddle.mean(loss_rpn_cls, name='loss_rpn_cls')

        # reg loss
        loc_tgt = paddle.cast(x=loc_tgt, dtype='float32')
        loc_tgt.stop_gradient = True
        loss_rpn_reg = ops.smooth_l1(
            input=deltas,
            label=loc_tgt,
            inside_weight=bbox_w,
            outside_weight=bbox_w,
            sigma=3.0, )
        loss_rpn_reg = paddle.sum(loss_rpn_reg)
        score_shape = paddle.shape(score_tgt)
        score_shape = paddle.cast(score_shape, dtype='float32')
        norm = paddle.prod(score_shape)
        norm.stop_gradient = True
        loss_rpn_reg = loss_rpn_reg / norm

        return {'loss_rpn_cls': loss_rpn_cls, 'loss_rpn_reg': loss_rpn_reg}
