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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from .. import ops


@register
class AnchorGenerator(object):
    __inject__ = ['anchor_generator', 'anchor_target_generator']

    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 stride=[16.0, 16.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 anchor_start_size=None):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.variance = variance
        self.anchor_start_size = anchor_start_size
        print('self.anchor_start_size ', self.anchor_start_size)

    def __call__(self, feats):
        anchors = []
        print('====self.anchor_start_size ', self.anchor_start_size)
        for level, rpn_feat in enumerate(feats):
            anchor_sizes = self.anchor_sizes if (
                self.anchor_start_size is None) else (self.anchor_start_size * 2
                                                      **level)
            stride = self.stride if (self.anchor_start_size is None) else (
                self.stride[0] * (2.**level), self.stride[1] * (2.**level))
            print(self.anchor_start_size, anchor_sizes, self.aspect_ratios,
                  stride)
            anchor, var = ops.anchor_generator(
                input=rpn_feat,
                anchor_sizes=anchor_sizes,
                aspect_ratios=self.aspect_ratios,
                stride=stride,
                variance=self.variance)
            anchors.append((anchor, var))
        return anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                For FPN models, `num_anchors` on every feature map is the same.
        """
        if self.anchor_start_size is None:
            return len(self.anchor_sizes) * len(self.aspect_ratios)
        else:
            if not isinstance(self.anchor_start_size, (list, tuple)):
                start_sizes = [self.anchor_start_size]
            return len(start_sizes) * len(self.aspect_ratios)
