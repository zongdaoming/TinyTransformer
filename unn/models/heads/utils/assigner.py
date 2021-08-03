import torch
import logging
import pdb

logger = logging.getLogger('global')

def get_rois_target_levels(levels, base_scale, rois):
    """
    Assign proposals to different level feature map to roi pooling

    Arguments:
        rois (FloatTensor): [R, 5] (batch_ix, x1, y1, x2, y2)
        levels (list of int): [L], levels. e.g.[2, 3, 4, 5, 6]
        base_scale: scale of the minimum level
    """
    w = rois[:, 3] - rois[:, 1] + 1
    h = rois[:, 4] - rois[:, 2] + 1
    scale = (w * h)**0.5
    eps = 1e-6
    target_levels = (scale / base_scale + eps).log2().floor()
    target_levels = target_levels.to(dtype=torch.int64)
    min_level, max_level = min(levels), max(levels)
    return torch.clamp(target_levels, min=min_level, max=max_level)


def map_rois_to_level(levels, base_scale, rois, original_inds=False):
    target_lvls = get_rois_target_levels(levels, base_scale, rois)
    rois_by_level, rois_ix_by_level = [], []
    for lvl in levels:
        ix = torch.nonzero(target_lvls == lvl).reshape(-1)
        rois_by_level.append(rois[ix])
        rois_ix_by_level.append(ix)
    map_from_inds = torch.cat(rois_ix_by_level)
    map_back_inds = torch.zeros((rois.shape[0], ), dtype=torch.int64, device=rois.device)
    seq_inds = torch.arange(rois.shape[0], device=rois.device)
    map_back_inds[map_from_inds] = seq_inds
    if original_inds:
        return rois_by_level, map_from_inds
    return rois_by_level, map_back_inds


def get_rois_by_level(levels, base_scale, rois):
    target_lvls = get_rois_target_levels(levels, base_scale, rois)
    rois_by_level, rois_ix_by_level = [], []
    for lvl in levels:
        ix = torch.nonzero(target_lvls == lvl).reshape(-1)
        rois_by_level.append(rois[ix])
        rois_ix_by_level.append(ix)
    return rois_by_level, rois_ix_by_level


def assign_to_levels(levels, base_scale, rois, *args):
    """
    Assign args to each level

    Arguments:
        rois (FloatTensor): [R, 5] (batch_ix, x1, y1, x2, y2)
        levels (list of int): [L], levels. e.g.[2, 3, 4, 5, 6]

    Returns:
        args: args(include rois) of each level
    """
    args_foreach_level = []
    rois_foreach_level, rois_ix_foreach_level = get_rois_by_level(levels, base_scale, rois)
    if len(args) == 0:
        return rois_foreach_level
    args_foreach_level = [[arg[ix] for ix in rois_ix_foreach_level] for arg in args]
    return [rois_foreach_level] + args_foreach_level


def test():
    levels = [0, 1, 2, 3]
    base_scale = 56
    rois = torch.FloatTensor([[0, 0, 0, 160, 160], [1, 0, 0, 1080, 1080], [2, 0, 0, 240, 240], [3, 20, 20, 400, 400],
                              [4, 4, 6, 80, 80]])
    _rois, recover_inds = map_rois_to_level(levels, base_scale, rois)
    _rois = torch.cat(_rois, dim=0)
    recover_rois = _rois[recover_inds]
    print(rois)
    print('*********')
    print(recover_rois)
    print('*********')
    print(_rois)
    print('*********')


if __name__ == '__main__':
    test()
