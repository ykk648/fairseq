# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import torch

def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    if torch.onnx.is_in_onnx_export():
        remainder = torch.ceil(m) * multiple - tsz
        remainder = remainder.type(torch.int64)
        pad_offset = (0,) * (-1 - dim) * 2
    else:
        remainder = math.ceil(m) * multiple - tsz
        if m.is_integer():
            return x, 0
        pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder
