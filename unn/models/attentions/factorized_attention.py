import torch
import torch.nn as nn
import torch.nn.functional as F
from .position import LearnedEmbedding


class FactorizedAttentionBlock(nn.Module):

    def __init__(self, inplanes, feat_planes, out_planes=None, kernel_size=1, stride=1, position_embedding=None,
                 **kwargs):

        super(FactorizedAttentionBlock, self).__init__()

        self.pe_use = False
        if position_embedding is not None:
            self.pe_use = True
            if position_embedding == "Learned Embedding":
                self.pe = LearnedEmbedding(inplanes)
            else:
                raise NotImplementedError()
        self.inplanes = inplanes
        self.feat_planes = feat_planes
        self.basis = nn.Conv2d(inplanes, feat_planes, kernel_size, padding=kernel_size // 2)
        self.coef = nn.Conv2d(inplanes, feat_planes, kernel_size, padding=kernel_size // 2)
        self.value = nn.Conv2d(inplanes, inplanes, kernel_size, padding=kernel_size // 2)
        self.final_use = False
        if out_planes is not None:
            self.final_use = True
            if stride == 1:
                self.final = nn.Conv2d(inplanes, feat_planes, 1)
            else:
                self.final = nn.Conv2d(inplanes, feat_planes, kernel_size=1, stride=2, padding=1)

    def forward(self, x):

        if self.pe_use:
            x = self.pe(x) + x

        B = x.shape[0]
        C = self.inplanes
        M = self.feat_planes
        H = x.shape[2]
        W = x.shape[3]

        # B*M*H*W
        basis = self.basis(x)
        basis = F.softmax(basis.view(-1, H * W), dim=1).view(B, M, H, W)

        # B*M*H*W
        coef = self.coef(x)
        coef = F.softmax(coef, dim=1)

        # B*C*H*W
        value = self.value(x)

        # B*HW*M
        basis = basis.view(B, M, -1)
        basis = torch.transpose(basis, 1, 2)

        # B*C*HW
        value = value.view(B, C, -1)

        # B * C * M
        res = torch.bmm(value, basis)

        # B * M * HW
        coef = coef.view(B, M, -1)

        # B * C * HW
        output = torch.bmm(res, coef)

        # B * C * H * W
        output = output.view(B, C, H, W)
        # output = self.bn(output)
        # output = self.up(output)
        output = output + x
        if self.final_use:
            output = self.final(output)
        return output


class FactorizedAttention(nn.Module):

    def __init__(self, **kwargs):
        super(FactorizedAttention, self).__init__()
        self.inplanes = kwargs['inplanes']
        self.out_planes = self.inplanes
        self.num_level = kwargs['num_level']
        for lvl_idx in range(self.num_level):
            if isinstance(self.inplanes, int):
                planes = self.inplanes
            else:
                planes = self.inplanes[lvl_idx]
            kwargs['inplanes'] = planes
            self.add_module(
                self.get_lateral_name(lvl_idx),
                FactorizedAttentionBlock(**kwargs)
            )

    def get_lateral_name(self, idx):
        return 'c{}_lateral'.format(idx)

    def get_lateral(self, idx):
        return getattr(self, self.get_lateral_name(idx))

    def get_outplanes(self):
        return self.out_planes

    def forward(self, input):

        features = input['features']
        assert self.num_level == len(features)
        laterals = [self.get_lateral(i)(x) for i, x in enumerate(features)]
        features = []
        for lvl_idx in range(self.num_level):
            out = laterals[lvl_idx]
            features.append(out)

        return {'features': features}
