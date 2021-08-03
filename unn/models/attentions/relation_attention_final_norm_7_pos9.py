import pdb
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unn.models.functions.embedding import ArrayEmbedding

logger = logging.getLogger('global')

class RelationAttention(nn.Module):

    def __init__(self, first_app_dim, second_app_dim, app_dim, first_type, second_type, head_count=1, app_mdim=1024):
        super(RelationAttention, self).__init__()
        self.head_count = head_count
        self.first_type = first_type
        self.second_type = second_type
        self.first_app_dim = first_app_dim
        self.second_app_dim = second_app_dim
        self.app_dim = app_dim
        self.app_mdim = app_mdim // head_count
        self.K = nn.Linear(self.app_dim * self.first_app_dim, app_mdim)
        self.Q = nn.Linear(self.app_dim * self.second_app_dim, app_mdim)
        self.V = nn.Linear(self.app_dim * self.first_app_dim, self.app_dim)
        self.G1 = nn.Linear(self.app_dim * self.first_app_dim, app_mdim)
        self.G2 = nn.Linear(self.app_dim * self.second_app_dim, app_mdim)
        self.relu = nn.ReLU(inplace=True)

    def preprocess(self, item, item_type, image_infos):
        '''
        convert the shape of appearance feature and position feature
        app: B, N, K
        pos: B, N, K 
        '''
        B = len(image_infos)
        if item_type == 'context':
            app, pos = item
            #item = self.conv(item)
            B, C, H, W = app.shape
            # B, C, HW
            app = app.view(B, C, -1).contiguous()
            # B, HW, C
            app = app.permute(0, 2, 1).contiguous()
            # B, C, HW
            pos = pos.view(B, C, -1).contiguous()
            # B, HW, C
            pos = pos.permute(0, 2, 1).contiguous()
            h = torch.arange(H).cuda() / H
            w = torch.arange(W).cuda() / W
            
        elif item_type == 'bbox':
            all_app, all_pos, rois = item
            app, pos = [], []
            for b_ix in range(B):
                inds = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
                if inds.shape[0] == 0:
                    app.append(None)
                    pos.append(None)
                    continue
                W, H = image_infos[b_ix][0], image_infos[b_ix][1]
                batch_app = all_app[inds]
                batch_pos = all_pos[inds]
                N = len(inds)
                batch_app = batch_app.view(N, -1).contiguous()
                batch_pos = batch_pos.view(N, -1).contiguous()
                app.append(batch_app)
                pos.append(batch_pos)
        else:
            raise ValueError('Unrecognized Type for Relation Attention Preprocess: ' + name)
        return app, pos

    def postprocess(self, item, item_type, image_infos):
        B = len(image_infos)
        if item_type == 'bbox':
            app = item
        else:
            raise ValueError('Unrecognized Type for Relation Attention Postprocess: ' + name)
        return app

    def calculate_geometry(self, first_pos, second_pos):
        pass

    def position_embedding(self, geometry, pos_mdim):
        N, C = geometry.shape
        geometry = geometry.view(-1, C)
        geometry = ArrayEmbedding(geometry, pos_mdim, dtype=geometry.dtype)
        geometry = geometry.view(N, -1)
        return geometry

    def forward(self, first, second, image_infos):
        B = len(image_infos)
        first_app, first_pos = self.preprocess(first, self.first_type, image_infos)
        second_app, second_pos = self.preprocess(second, self.second_type, image_infos)
        new_features = []
        for b_ix in range(B):
            # N, C1, 1
            batch_first_app = first_app[b_ix]
            batch_first_pos = first_pos[b_ix]
            # M, C2, 1
            batch_second_app = second_app[b_ix]
            batch_second_pos = second_pos[b_ix]
            if batch_first_app is None or batch_second_app is None:
                new_features.append(None)
                continue
            N = batch_first_app.shape[0]
            M = batch_second_app.shape[0]
            G1_feature = self.G1(batch_first_pos).view(1, -1, self.head_count, self.app_mdim)
            G1_feature = G1_feature.permute(2, 0, 1, 3).contiguous()
            G2_feature = self.G2(batch_second_pos).view(-1, 1, self.head_count, self.app_mdim)
            G2_feature = G2_feature.permute(2, 0, 1, 3).contiguous()

            # 1, N, HC, C / HC
            V_feature = self.V(batch_first_app).view(1, N, self.head_count, self.app_dim // self.head_count)
            # HC, 1, N, C / HC
            V_feature = V_feature.permute(2, 0, 1, 3).contiguous()
            # M, 1, HC, C / HC
            Q_feature = self.Q(batch_second_app).view(M, 1, self.head_count, self.app_mdim)
            # HC, M, 1, C / HC
            Q_feature = Q_feature.permute(2, 0, 1, 3).contiguous()
            Q_norm = torch.norm(Q_feature, dim=3).detach().view(self.head_count, M, 1, 1)
            Q_feature = Q_feature / Q_norm
            G2_norm = torch.norm(G2_feature, dim=3).detach().view(self.head_count, M, 1, 1)
            G2_feature = G2_feature / G2_norm

            # 1, N, HC, C / HC
            K_feature = self.K(batch_first_app).view(1, N, self.head_count, self.app_mdim)
            # HC, 1, N, C / HC
            K_feature = K_feature.permute(2, 0, 1, 3).contiguous()
            
            # HC, M, N
            G_feature = torch.sum(G1_feature * G2_feature, dim=3) / self.app_mdim
            # HC, M, N
            A_feature = torch.sum(Q_feature * K_feature, dim=3) / self.app_mdim

            # HC, M, N
            weight = torch.exp(A_feature * G_feature)
            # HC, M, N
            weight = weight / torch.sum(weight, dim=2).view(self.head_count, M, 1)
            # HC, M, N, 1
            weight = weight.reshape(self.head_count, M, N, 1)
            # HC, M, N, C / HC
            V_feature = torch.sum(V_feature * weight, dim=2)
            # M, HC, C / HC
            V_feature = V_feature.permute(1, 0, 2).contiguous()
            # M, C
            V_feature = V_feature.view(M, self.app_dim)
            new_features.append(V_feature)

        new_second_feature = self.postprocess(new_features, self.second_type, image_infos)
        return new_second_feature


class CBRelationAttention(RelationAttention):
    '''
    Context and bbox
    '''
    def __init__(self, first_app_dim, second_app_dim, app_dim, head_count=1):
        super(CBRelationAttention, self).__init__(first_app_dim, second_app_dim, app_dim, 'context', 'bbox', head_count, app_dim)

class BBRelationAttention(RelationAttention):
    '''
    Bbox and bbox
    '''
    def __init__(self, first_app_dim, second_app_dim, app_dim, head_count=1, app_mdim=1024):
        super(BBRelationAttention, self).__init__(first_app_dim, second_app_dim, app_dim, 'bbox', 'bbox', head_count, app_mdim)

class MultiRelationAttention(nn.Module):

    def __init__(self, cfg):
        super(MultiRelationAttention, self).__init__()
        if cfg.get('bb_attention', None) is not None:
            tmp_cfg = cfg['bb_attention']
            self.bbox = BBRelationAttention(**tmp_cfg)
        else:
            self.bbox = None
        if cfg.get('cb_attention', None) is not None:
            tmp_cfg = cfg['cb_attention']
            self.context = CBRelationAttention(**tmp_cfg)
        else:
            self.context = None

    def forward(self, cfg):
        ori = cfg['rois_feature']
        if self.bbox is not None:
            new_features = self.bbox((cfg['rois_feature'], cfg['rois_embed'], cfg['rois']), (cfg['rois_feature'], cfg['rois_embed'], cfg['rois']), cfg['image_info'])
            B = len(cfg['image_info'])
            rois = cfg['rois']
            for b_ix in range(B):
                inds = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
                if len(inds) == 0: continue
                batch_feature = new_features[b_ix].contiguous()
                N, C = batch_feature.shape[:2]
                ori[inds] = ori[inds] * F.sigmoid(batch_feature.view(N, C, 1))
        if self.context is not None:
            new_features = self.context((cfg['context'], cfg['embed']), (cfg['rois_feature'], cfg['rois_embed'], cfg['rois']), cfg['image_info'])
            B = len(cfg['image_info'])
            rois = cfg['rois']
            for b_ix in range(B):
                inds = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
                if len(inds) == 0: continue
                batch_feature = new_features[b_ix].contiguous()
                N, C = batch_feature.shape[:2]
                ori[inds] = ori[inds] * F.sigmoid(batch_feature.view(N, C, 1))
        return ori

