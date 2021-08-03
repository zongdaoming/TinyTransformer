import torch
import numpy as np

# map of (anchor_stride, feature_stride) to anchors-over-plane
_ANCHOR_BUFFER = {}


def get_anchors_over_plane(featmap_h,
                           featmap_w,
                           featmap_stride,
                           anchor_ratios,
                           anchor_scales,
                           anchor_stride,
                           dtype=None,
                           device=None):
    anchors = build_anchor_from_scratch(
        featmap_h, featmap_w, featmap_stride, anchor_ratios, anchor_scales, anchor_stride, dtype=dtype, device=device)
    return anchors.reshape(-1, 4)

    key = (anchor_stride, featmap_stride)
    if key in _ANCHOR_BUFFER:
        anchor_buf = _ANCHOR_BUFFER[key]
        buf_h, buf_w = anchor_buf.shape[0:2]
        if buf_w >= featmap_w and buf_h >= featmap_h:
            anchors = anchor_buf[:featmap_h, :featmap_w, :, :]
            return anchors.reshape(-1, 4)
    _ANCHOR_BUFFER[key] = build_anchor_from_scratch(
        featmap_h, featmap_w, featmap_stride, anchor_ratios, anchor_scales, anchor_stride, dtype=dtype, device=device)
    return _ANCHOR_BUFFER[key].reshape(-1, 4)


def build_anchor_from_scratch(featmap_h,
                              featmap_w,
                              featmap_stride,
                              anchor_ratios,
                              anchor_scales,
                              anchor_stride,
                              dtype=None,
                              device=None):
    # [A, 4], anchors over one pixel
    anchors_overgrid = get_anchors_over_grid(anchor_ratios, anchor_scales, anchor_stride)
    anchors_overgrid = torch.from_numpy(anchors_overgrid).to(device=device, dtype=dtype)
    # spread anchors over each grid
    shift_x = torch.arange(0, featmap_w * featmap_stride, step=featmap_stride, dtype=dtype, device=device)
    shift_y = torch.arange(0, featmap_h * featmap_stride, step=featmap_stride, dtype=dtype, device=device)
    # [featmap_h, featmap_w]
    shift_y, shift_x = torch.meshgrid((shift_y, shift_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

    anchors_overgrid = anchors_overgrid.reshape(1, -1, 4)
    shifts = shifts.reshape(-1, 1, 4).to(anchors_overgrid)
    anchors_overplane = anchors_overgrid + shifts
    return anchors_overplane.reshape(featmap_h, featmap_w, -1, 4)


def get_anchors_over_grid(ratios, scales, stride):
    """
    generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    ratios = np.array(ratios)
    scales = np.array(scales)
    anchor = np.array([1, 1, stride, stride], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
    return anchors


def _ratio_enum(anchor, ratios):
    """enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    """return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


if __name__ == '__main__':
    ratios = [0.5, 1, 2]
    scales = [2, 4, 8, 16, 32]
    stride = 16
    feat_h, feat_w, feat_s = 5, 3, 16
    anchors = get_anchors_over_plane(feat_h, feat_w, feat_s, ratios, scales, stride)
    anchors = anchors.cpu().numpy()
    print(anchors)
    print(anchors.shape)
