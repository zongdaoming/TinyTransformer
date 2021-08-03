import numpy as np
import torch

from .._C import naive_nms as nms
from .._CY import soft_nms  # noqa: F401


def naive_nms(dets, thresh):
    assert dets.shape[1] == 5
    keep = torch.LongTensor(dets.shape[0])
    num_out = torch.LongTensor(1)
    if torch.cuda.is_available():
        nms.gpu_nms(keep, num_out, dets.cuda().float(), thresh)
    else:
        dets = dets.cpu()
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.from_numpy(np.arange(dets.shape[0])).long()
        nms.cpu_nms(keep, num_out, dets.float(), order, areas, thresh)
    return keep[:num_out[0]].contiguous().to(device=dets.device)
