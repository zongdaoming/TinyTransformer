import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .... import extensions as E
from . import accuracy as A

logger = logging.getLogger('global')


def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret


def l1_loss(input, target, scale_type='linear', reduction='mean', normalizer=None):
    if scale_type == 'linear':
        input = input
        target = target
    elif scale_type == 'log':
        input = torch.log(input)
        target = torch.log(target)
    else:
        raise NotImplementedError
    loss = torch.abs(input - target)
    loss = _reduce(loss, reduction=reduction, normalizer=normalizer)
    return loss


def balanced_l1_loss(input, target, sigma=1.0, alpha=0.5, gamma=1.5, reduction='mean', normalizer=None):
    beta = 1. / (sigma**2)
    diff = torch.abs(input - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b * (b * diff + 1)
        * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)
    loss = _reduce(loss, reduction, normalizer=normalizer)
    return loss


def smooth_l1_loss(input, target, sigma, reduce=True, normalizer=1.0):
    beta = 1. / (sigma**2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer

def cross_entropy_weight(input, target, sample_weight=None, cls_weight=None):
    if sample_weight is None:
        return F.cross_entropy(input, target, weight=cls_weight)
    sample_num = target.size()[0]
    log_input = F.log_softmax(input, 1)
    loss = F.nll_loss(log_input * sample_weight.reshape(sample_num, 1),
                target, cls_weight)
    return loss * sample_num / sample_weight.sum()  # normal

def ohem_loss(batch_size, cls_pred, cls_target, loc_pred, loc_target, cls_type='softmax', smooth_l1_sigma=1.0):
    """
    Arguments:
        batch_size (int): number of sampled rois for bbox head training
        loc_pred (FloatTensor): [R, 4], location of positive rois
        loc_target (FloatTensor): [R, 4], location of positive rois
        pos_mask (FloatTensor): [R], binary mask for sampled positive rois
        cls_pred (FloatTensor): [R, C]
        cls_target (LongTensor): [R]

    Returns:
        cls_loss, loc_loss (FloatTensor)
    """
    if cls_type == 'softmax':
        ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    else:
        ohem_cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='none')
    if loc_pred is None:
        ohem_loc_loss = torch.zeros_like(ohem_cls_loss)
    else:
        ohem_loc_loss = smooth_l1_loss(loc_pred, loc_target, sigma=smooth_l1_sigma, reduce=False)
    loss = ohem_cls_loss + ohem_loc_loss

    sorted_ohem_loss, idx = torch.sort(loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], batch_size)
    if keep_num <= sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]

    cls_loss = ohem_cls_loss.sum() / keep_num
    loc_loss = ohem_loc_loss.sum() / keep_num
    return cls_loss, loc_loss, keep_idx_cuda


def get_rpn_cls_loss(cls_pred, cls_target, sample_cls_mask, loss_type):
    """
    Arguments:
        cls_pred (FloatTensor): [B*K, C]
        cls_target (LongTensor): [B*K]
        sample_cls_mask (ByteTensor): [B, K], binary mask for sampled rois
        loss_type (str): sigmoid or softmax

    Returns:
        cls_loss, acc (FloatTensor)
    """
    sample_cls_mask = sample_cls_mask.reshape(-1)

    if loss_type == "softmax":
        cls_pred = cls_pred.reshape(cls_target.numel(), -1)
        cls_target = cls_target.reshape(-1)
        cls_loss = F.cross_entropy(cls_pred, cls_target.long(), ignore_index=-1)
        acc = A.accuracy(cls_pred, cls_target)[0]
    elif loss_type == "sigmoid":
        cls_pred = cls_pred.reshape(-1)
        cls_target = cls_target.reshape(-1)

        normalizer = (sample_cls_mask > 0).float().sum()
        normalizer = max(1, normalizer.item())
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target.float(), reduction='none')
        cls_loss = (cls_loss * sample_cls_mask.float()).sum() / normalizer
        # acc = torch.tensor([0]).cuda().float()  # for sigmoid, there is a bug in A.accuracy
        acc = A.binary_accuracy(cls_pred, cls_target)[0]
    return cls_loss, acc


def get_rpn_loc_loss(loc_pred, loc_target, sample_loc_mask, sigma, normalizer):
    """
    Arguments:
        loc_pred (FloatTensor): [B*K, 4]
        loc_target (LongTensor): [B*K, 4]
        sample_loc_mask (ByteTensor): [B, K], binary mask for sampled poitive rois

    Returns:
        loc_loss (FloatTensor)
    """
    sample_loc_mask = sample_loc_mask.reshape(-1)
    loc_target = loc_target.reshape(-1, 4)[sample_loc_mask]
    loc_pred = loc_pred.reshape(-1, 4)[sample_loc_mask]
    loc_loss = smooth_l1_loss(loc_pred, loc_target, sigma, normalizer=normalizer)
    return loc_loss


def get_focal_loss(cls_pred, cls_target, normalizer, num_classes, cfg_loss):
    """
    Arguments:
        cls_pred (FloatTensor): [B*K, C]
        cls_target (LongTensor): [B*K]
        cfg_loss: config for focal loss

    Returns:
        cls_loss, acc (FloatTensor)
    """
    alpha = cfg_loss['alpha']
    gamma = cfg_loss['gamma']
    loss_type = cfg_loss['type']
    C = {'sigmoid': -1, 'softmax': 0}[loss_type] + num_classes

    cls_pred = cls_pred.float().view(-1, C)
    cls_target = cls_target.int().view(-1)
    normalizer = torch.cuda.FloatTensor([normalizer])

    loss_fn = {'sigmoid': E.SigmoidFocalLossFunction, 'softmax': E.SoftmaxFocalLossFunction}[loss_type]

    loss_fn = loss_fn(gamma, alpha, C)
    cls_loss = loss_fn(cls_pred, cls_target, normalizer)

    if loss_type == 'softmax':
        acc = A.accuracy(cls_pred, cls_target.long())[0]
    elif loss_type == 'sigmoid':
        acc = A.accuracy(cls_pred, cls_target.long() - 1, ignore_indices=[-1, -2])[0]
    else:
        raise NotImplementedError('{} is not supported for focal loss'.format(loss_type))
    return cls_loss, acc


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class GHMCLoss(nn.Module):
    def __init__(self, bins=10, momentum=0, loss_weight=1.0):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        self.loss_weight = loss_weight
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def binarize_target(self, input, target):
        """ convert target index to one-hot target
        Args:
            input: [B, A, C]
            target: [B, A]
        Returns:
            target: [B, A, C]
            mask: [B, A, C]
        """
        binary_targets = torch.zeros_like(input)
        mask = torch.zeros_like(input)
        pos_inds = target > 0
        cls_inds = target[pos_inds] - 1
        binary_targets[pos_inds, cls_inds] = 1
        mask[target > -1, :] = 1
        return binary_targets, mask

    def forward(self, input, target, mlvl_shapes=None):
        """ Args:
        input [batch_num, anchor_num, C]:
            The direct prediction of classification fc layer.
        target [batch_num, anchor_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        target, mask = self.binarize_target(input, target)

        if mlvl_shapes is None:
            return self.forward_single(input, target, mask)

        mlvl_size = [_[-1] for _ in mlvl_shapes]
        assert input.ndimension() == 3
        assert target.ndimension() == 3
        assert mask.ndimension() == 3

        inputs = input.split(mlvl_size, dim=1)
        targets = target.split(mlvl_size, dim=1)
        masks = mask.split(mlvl_size, dim=1)

        loss = 0
        for i, t, m in zip(inputs, targets, masks):
            loss += self.forward_single(i, t, m)
        return loss

    def forward_single(self, input, target, mask):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        valid = mask > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(input, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

    @classmethod
    def from_params(cls, params):
        bins = params['bins']
        momentum = params['momentum']
        loss_weight = params['loss_weight']
        return cls(bins, momentum, loss_weight)


class GHMRLoss(nn.Module):
    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0):
        super(GHMRLoss, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        self.loss_weight = loss_weight
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def forward(self, input, target, mask, mlvl_shapes=None):
        """ Args:
        input [batch_num, anchor_num, 4]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, anchor_num, 4]:
            The target regression values with the same size of input.
        mask [batch_num, anchor_num]: mask for each anchor
        """
        # expand to each coordinate
        mask = mask.float().reshape(input.shape[0], input.shape[1], 1).repeat(1, 1, 4)
        if mlvl_shapes is None:
            return self.forward_single(input, target, mask)

        mlvl_size = [_[-1] for _ in mlvl_shapes]
        assert input.ndimension() == 3
        assert target.ndimension() == 3
        assert mask.ndimension() == 3

        inputs = input.split(mlvl_size, dim=1)
        targets = target.split(mlvl_size, dim=1)
        masks = mask.split(mlvl_size, dim=1)

        loss = 0
        for i, t, m in zip(inputs, targets, masks):
            loss += self.forward_single(i, t, m)
        return loss

    def forward_single(self, input, target, mask):
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = input - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = mask > 0
        tot = max(mask.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight

    @classmethod
    def from_params(cls, params):
        mu = params['mu']
        bins = params['bins']
        momentum = params['momentum']
        loss_weight = params['loss_weight']
        return cls(mu, bins, momentum, loss_weight)
