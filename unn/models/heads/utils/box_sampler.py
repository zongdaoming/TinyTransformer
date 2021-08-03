import torch


def sample_fixed(positive, negative, batch_size, pos_percent):
    """
    Sample pos_percent positives from `positive` if not empty,
    so does to negative
    """
    expected_pos_num = int(batch_size * pos_percent)
    expected_neg_num = batch_size - expected_pos_num
    if positive.numel() > 0:
        if expected_pos_num > positive.numel():
            shuffle_pos = torch.randperm(expected_pos_num, device=positive.device) % positive.numel()
        else:
            shuffle_pos = torch.randperm(positive.numel(), device=positive.device)[:expected_pos_num]
        positive = positive[shuffle_pos]
    if negative.numel() > 0:
        if expected_neg_num > negative.numel():
            shuffle_neg = torch.randperm(expected_neg_num, device=negative.device) % negative.numel()
        else:
            shuffle_neg = torch.randperm(negative.numel(), device=negative.device)[:expected_neg_num]
        negative = negative[shuffle_neg]
    return positive, negative


def sample_flexible(positive, negative, batch_size, pos_percent):
    """
    Sample positives up to pos_percent, if positive num is not enough,
    sample more negatives to make it up to a batch_size
    """
    expected_pos_num = int(batch_size * pos_percent)
    if positive.numel() > expected_pos_num:
        shuffle_pos = torch.randperm(positive.numel(), device=positive.device)[:expected_pos_num]
        positive = positive[shuffle_pos]

    expected_neg_num = batch_size - positive.numel()
    if negative.numel() > expected_neg_num:
        shuffle_neg = torch.randperm(negative.numel(), device=negative.device)[:expected_neg_num]
        negative = negative[shuffle_neg]
    return positive, negative


def sample(match_target, cfg):
    """
    Arguments:
        match_target (LongTensor): output of `match` function
        cfg['batch_size'] (int): total number of samples for training

    Returns:
        positive_inds, negative_inds (LongTensor): indices of sampled rois
    """
    pos_inds = torch.nonzero(match_target >= 0).reshape(-1)
    neg_inds = torch.nonzero(match_target == -1).reshape(-1)
    if cfg['type'] == 'naive':
        pos_inds, neg_inds = sample_flexible(pos_inds, neg_inds, cfg['batch_size'], cfg['positive_percent'])
    elif cfg['type'] == 'force_keep_ratio':
        pos_inds, neg_inds = sample_fixed(pos_inds, neg_inds, cfg['batch_size'], cfg['positive_percent'])
    elif cfg['type'] == 'keep_all':
        # for ohem and focal loss, keep all samples
        pass
    else:
        raise NotImplementedError('{} not supported'.format(cfg['type']))

    return pos_inds, neg_inds
