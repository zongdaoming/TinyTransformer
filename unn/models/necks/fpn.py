import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import initialize_from_cfg

__all__ = ['FPN']


class FPN(nn.Module):
    """
    Note:
        If num_level is larger than backbone's output feature layers, additional layers will be stacked

    Arguments:
        inplanes (list): input channel
        outplanes (int): output channel, all layers are the same
        start_level (int): start layer of backbone to apply FPN
        num_level (int): number of FPN layers
        out_strides (list): stride of FPN output layers
        downsample (str): method to downsample, for FPN, it's pool, for RetienaNet, it's conv
        upsample (str): method to upsample, nearest or bilinear
        initializer (dict): config for model parameter initialization
    """

    def __init__(self,
                 inplanes,
                 outplanes,
                 start_level,
                 num_level,
                 out_strides,
                 downsample,
                 upsample,
                 tocaffe_friendly=False,
                 freeze=False,
                 initializer=None,
                 cfg=None):
        super(FPN, self).__init__()

        assert downsample in ['pool', 'conv'], downsample
        assert isinstance(inplanes, list)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.outstrides = out_strides
        self.start_level = start_level
        self.num_level = num_level
        self.downsample = downsample
        self.upsample = upsample
        self.tocaffe_friendly = tocaffe_friendly
        self.freeze = freeze
        if cfg is None:
            self.cfg = {}
        else:
            self.cfg = cfg
        self.freeze = self.cfg.get('freeze', False)
        assert num_level == len(out_strides)

        for lvl_idx in range(num_level):
            if lvl_idx < len(inplanes):
                planes = inplanes[lvl_idx]
                self.add_module(
                    self.get_lateral_name(lvl_idx),
                    nn.Conv2d(planes, outplanes, 1))
                self.add_module(
                    self.get_pconv_name(lvl_idx),
                    nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1))
            else:
                if self.downsample == 'pool':
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        nn.MaxPool2d(kernel_size=1, stride=2, padding=0))  # strange pooling
                else:
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=2, padding=1))
        initialize_from_cfg(self, initializer)
        if self.freeze:
            for name, module in self.named_children():
                for param in module.parameters():
                    param.requires_grad = False

    def get_lateral_name(self, idx):
        return 'c{}_lateral'.format(idx + self.start_level)

    def get_lateral(self, idx):
        return getattr(self, self.get_lateral_name(idx))

    def get_downsample_name(self, idx):
        return 'p{}_{}'.format(idx + self.start_level, self.downsample)

    def get_downsample(self, idx):
        return getattr(self, self.get_downsample_name(idx))

    def get_pconv_name(self, idx):
        return 'p{}_conv'.format(idx + self.start_level)

    def get_pconv(self, idx):
        return getattr(self, self.get_pconv_name(idx))

    def real_forward(self, input):
        """
        Note:
            For faster-rcnn, get P3-P6 from C2-C5, then P7 = pool(P6)
            For RetinaNet, get P3-P5 from C3-C5, then P6 = Conv(C5), P7 = Conv(P6)

        Arguments:
            input['features'] (list): pyramid features,
            input['strides'] (list): strides of pyramid features
        """
        features = input['features']
        assert len(self.inplanes) == len(features)
        laterals = [self.get_lateral(i)(x) for i, x in enumerate(features)]
        features = []

        # top down pathway
        for lvl_idx in range(len(self.inplanes))[::-1]:
            if lvl_idx < len(self.inplanes) - 1:
                if self.tocaffe_friendly:
                    laterals[lvl_idx] += F.interpolate(
                        laterals[lvl_idx + 1], scale_factor=2, mode=self.upsample)
                else:
                    # nart_tools may not support to interpolate to the size of other feature
                    # you may need to modify upsample or interp layer in prototxt manually.
                    upsize = laterals[lvl_idx].shape[-2:]
                    laterals[lvl_idx] += F.interpolate(
                        laterals[lvl_idx + 1], size=upsize, mode=self.upsample)
            out = self.get_pconv(lvl_idx)(laterals[lvl_idx])
            features.append(out)
        features = features[::-1]

        # bottom up further
        if self.downsample == 'pool':
            x = features[-1]  # for faster-rcnn, use P6 to get P7
        else:
            x = laterals[-1]  # for RetinaNet, ues C5 to get P6, P7
        for lvl_idx in range(self.num_level):
            if lvl_idx >= len(self.inplanes):
                x = self.get_downsample(lvl_idx)(x)
                features.append(x)
        return {'features': features, 'strides': self.get_outstrides()}

    def forward(self, input):
        if self.freeze:
            with torch.no_grad():
                return self.real_forward(input)
        else:
            return self.real_forward(input)

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
