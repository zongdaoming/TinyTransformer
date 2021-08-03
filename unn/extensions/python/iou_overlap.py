from .._C import overlaps


def gpu_iou_overlap(b1, b2):
    if b1.numel() == 0 or b2.numel() == 0:
        return b1.new_zeros((0,))

    assert b1.shape[1] >= 4 and b2.shape[1] >= 4
    assert b1.is_cuda and b2.is_cuda

    b1 = b1[:, :4].contiguous()
    b2 = b2[:, :4].contiguous()
    ious = b1.new_zeros((b1.shape[0], b2.shape[0]))
    overlaps.iou(b1, b2, ious)
    return ious
