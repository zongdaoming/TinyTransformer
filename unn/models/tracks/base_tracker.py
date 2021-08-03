import numpy as np
import torch.nn as nn
import torch
import logging
import pdb
from torch.autograd import Variable
from .km_match import KMMatch

logger = logging.getLogger('global')

def to_np_array(x):
    if x is None:
        return x
    if isinstance(x, Variable): x = x.data
    return x.cpu().float().numpy() if torch.is_tensor(x) else x

class DetectionRecord:

    def __init__(self, cx, cy, h, w, score, label):
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w
        self.score = score
        self.label = label

class BaseTrackRecord:

    def __init__(self, track_id, roi, track_score, cur_frame):
        '''
        Input:
            track_id: int
            roi: np.array [7] batch_ix, x1, y1, x2, y2, score, cls
        '''
        x1, y1, x2, y2, score, cls = roi[1:]
        self.obj = DetectionRecord((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1, score, cls)
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.h = x2 - x1
        self.w = y2 - y1
        self.type = 1   # high score track / low score track
        self.id = track_id
        self.track_score = track_score
        self.track_len = 1
        self.last_update = cur_frame
        self.end_frame = cur_frame

    def update(self, roi, cur_frame):
        x1, y1, x2, y2, score, cls = roi[1:]
        self.obj = DetectionRecord((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1, score, cls)
        self.last_update = cur_frame
        self.end_frame = cur_frame


class BaseTracker(nn.Module):

    def __init__(self, match_thresh=0.4, type_thresh=0.5, init_score=0.75, frame_skip=1, drop_len=40, decay_rate=0.85, num_classes=10):
        super(BaseTracker, self).__init__()
        self.reset()
        self.num_classes = num_classes
        self.match_thresh = self.tolist(match_thresh, self.num_classes)
        self.type_thresh = self.tolist(type_thresh, self.num_classes)
        self.init_score = self.tolist(init_score, self.num_classes)
        self.frame_skip = self.tolist(frame_skip, self.num_classes)
        self.drop_len = self.tolist(drop_len, self.num_classes)
        self.decay_rate = self.tolist(decay_rate, self.num_classes)
        self.cur_frame = 0
    
    def tolist(self, instance, num_classes):
        if isinstance(instance, list):
            return instance
        elif isinstance(instance, float) or isinstance(instance, int):
            return [instance for _ in range(num_classes)]
        else:
            return None

    def reset(self):
        self.track_list = []
        self.isolate = []
        self.track_id_count = 0


    def new_id(self):
        '''
        Function:
            return a unique track_id for new object

        Output:
            self.track_id_count: int
        '''

        self.track_id_count += 1
        return int(self.track_id_count)

    def generate_score_mat(self, track_list, rois):
        
        N = len(track_list)
        M = rois.shape[0]
        S = max(N, M)
        score_mat = np.zeros((S, S))

        for i in range(N):
            trk = track_list[i]
            for j in range(M):
                roi = rois[j]
                det_cx = (roi[1] + roi[3]) / 2.0
                det_cy = (roi[2] + roi[4]) / 2.0
                det_h = roi[3] - roi[1]
                det_w = roi[4] - roi[2]
                app_sim = 1.0
                mot_sim = np.exp(-0.5 * (np.power((trk.cx - det_cx) / det_h, 2.0) + np.power((trk.cy - det_cy) / det_w, 2.0)))
                shp_sim = np.exp(-1.5 * (abs(trk.h - det_h) / (trk.h + det_h) + abs(trk.w - det_w) / (trk.w + det_w)))

                score_mat[i][j] = app_sim * mot_sim * shp_sim
                if int(roi[6]) != int(trk.obj.label):
                    score_mat[i][j] = 0
        return score_mat


    def associate(self, rois):

        high_trk = []
        high_idx = []

        low_trk = []
        low_idx = []
        for i in range(len(self.track_list)):
            if self.track_list[i].type == 1:
                high_trk.append(self.track_list[i])
                high_idx.append(i)
            else:
                low_trk.append(self.track_list[i])
                low_idx.append(i)
        M = rois.shape[0]
        isolate = np.ones((M))
        self.track_id = np.zeros((M), dtype=np.int32)
        if len(high_trk) > 0:
            score_mat = self.generate_score_mat(high_trk, rois)
            N = len(high_trk)
            S = score_mat.shape[0]
            match = KMMatch(S, S, score_mat).match
            for i in range(M):
                if match[i] == -1:
                    continue
                if match[i] >= N:
                    continue
                ind = high_idx[match[i]]
                label = int(rois[i][6])
                if score_mat[match[i]][i] > self.match_thresh[label]:
                    self.track_list[ind].track_score = (self.track_list[ind].track_score * self.track_list[ind].track_len + score_mat[match[i]][i]) / (self.track_list[ind].track_len + 1)
                    self.track_list[ind].track_len += 1
                    self.track_list[ind].update(rois[i], self.cur_frame)
                    isolate[i] = 0
                    self.track_id[i] = self.track_list[ind].id
                else:
                    self.track_list[ind].track_len = 0

        urois = []
        uidx = []
        for i in range(M):
            if isolate[i] == 1:
                urois.append(rois[i])
                uidx.append(i)

        urois = np.array(urois)
        if len(low_trk) > 0 and urois.shape[0] > 0:
            score_mat = self.generate_score_mat(low_trk, urois)
            S = score_mat.shape[0]
            uN = len(low_trk)
            uM = urois.shape[0]
            match = KMMatch(S, S, score_mat).match
            for i in range(uM):
                if match[i] == -1:
                    continue
                if match[i] >= uN:
                    continue
                ind = low_idx[match[i]]
                label = int(rois[i][6])
                if score_mat[match[i]][i] > self.match_thresh[label]:
                    self.track_list[ind].track_score = (self.track_list[ind].track_score * self.track_list[ind].track_len + score_mat[match[i]][i]) / (self.track_list[ind].track_len + 1)
                    self.track_list[ind].track_len += 1
                    self.track_list[ind].update(urois[i], self.cur_frame)
                    isolate[uidx[i]] = 0
                    self.track_id[uidx[i]] = self.track_list[ind].id
                else:
                    self.track_list[ind].track_len = 0
        self.isolate = isolate


    def update(self):

        alpha = 1.2
        for i in range(len(self.track_list)):
            label = int(self.track_list[i].obj.label)
            if self.track_list[i].last_update == self.cur_frame:
                self.track_list[i].track_score = self.track_list[i].track_score * (1 - np.exp(-alpha * np.sqrt(self.track_list[i].track_len)))
            elif self.track_list[i].last_update <= self.cur_frame - self.frame_skip[label]:
                self.track_list[i].track_score = self.track_list[i].track_score * self.decay_rate[label]

        N = len(self.track_list)
        reserve = np.ones((N))
        for i in range(N):
            label = int(self.track_list[i].obj.label)
            if self.track_list[i].type == 1 and self.track_list[i].track_score < self.type_thresh[label]:
                self.track_list[i].type = 0
                self.track_list[i].end_frame = self.cur_frame
            elif self.track_list[i].type == 0:
                if self.track_list[i].track_score > self.type_thresh[label]:
                    self.track_list[i].type = 1
                if (self.cur_frame - self.track_list[i].end_frame >= self.drop_len[label]):
                    reserve[i] = 0

        new_track_list = []
        for i in range(N):
            if reserve[i] == 1:
                new_track_list.append(self.track_list[i])

        self.track_list = new_track_list
        N = len(self.track_list)
        for i in range(N):
            self.track_list[i] = self.status_update(self.track_list[i])



    def status_update(self, trk):
        pass

    def generate_track(self, roi, track_id, track_score, cur_frame):
        pass

    def process(self, rois, cur_frame):
        '''
        Function:
            For each rois, return a unique track id

        Input:
            rois: cuda.FloatTensor [N, 7] batch_ix, x1, y1, x2, y2, score, cls
            cur_frame: int

        Output:
            track_id: np.array [N, 1] idx
        '''
        rois = to_np_array(rois)
        self.cur_frame = cur_frame
        self.associate(rois)
        self.update()
        M = rois.shape[0]
        for i in range(M):
            label = int(rois[i][6])
            if self.isolate[i] == 1:
                self.track_id[i] = self.new_id()
                self.track_list.append(self.generate_track(rois[i], self.track_id[i], self.init_score[label], self.cur_frame))

        return self.track_id
