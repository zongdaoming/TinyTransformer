import torch
from torch.autograd import Variable
import numpy as np
import logging
import pdb

logger = logging.getLogger('global')

def to_np_array(x):
    if x is None:
        return x
    if isinstance(x, Variable): x = x.data
    return x.cpu().float().numpy() if torch.is_tensor(x) else x

class SingleFramePostProcessor:

    def __init__(self):
        pass

    def process(self, input):
        pass

class SingleFramePostProcessorFactory:

    @classmethod
    def create(cls, cfg):

        name = cfg['name']
        kwargs = cfg['kwargs']
        if name == 'none':
            processor = EmptyProcessor(**kwargs)
        elif name == 'greedy':
            processor = GreedyProcessor(**kwargs)
        elif name == 'greedy_face2body':
            processor = GreedyProcessorFace2Body(**kwargs)
        else:
            raise ValueError('Unrecognized Single Frame Post Processor: ' + name)

        return processor


class EmptyProcessor(SingleFramePostProcessor):

    def __init__(self):
        pass

    def process(self, input):

        return input

class GreedyProcessorFace2Body(SingleFramePostProcessor):

    def __init__(self, connection_limits):
        self.clim = connection_limits

    def process(self, input):

        rois = input['dt_bboxes']
        assos = input['dt_assos']
        rois, assos = map(to_np_array, [rois, assos])
        
        C = len(self.clim)
        N = rois.shape[0]
        M = assos.shape[0]
        if 'use_time' not in input:
            use_time = [[0 for _ in range(N)] for __ in range(C)]
        else:
            use_time = input['use_time']
        pos_to_id = {}
        new_assos = []
        for i in range(N):
            box = rois[i]
            pos = (box[1], box[2], box[3], box[4])
            pos_to_id[pos] = i

        order = np.argsort(assos[:, 12].reshape(-1))
        order = order[::-1]
        assos = assos[order]
        assos_face = assos[assos[:, 10] == 2]

        matched_body_idx = []
        for i, asso in enumerate(assos_face):
            cls = int(asso[11])
            if cls == 0:
                continue
            posx = (asso[1], asso[2], asso[3], asso[4])
            posy = (asso[6], asso[7], asso[8], asso[9])
            idx = pos_to_id[posx]
            idy = pos_to_id[posy]
            clsx = int(asso[5])
            clsy = int(asso[10])
            if clsx == 0 or clsy == 0:
                continue
            if use_time[cls - 1][idx] < self.clim[cls - 1][clsx - 1] and use_time[cls - 1][idy] < self.clim[cls - 1][clsy - 1]:
                matched_body_idx.append(idx)
                new_assos.append(asso)
                use_time[cls - 1][idx] += 1
                use_time[cls - 1][idy] += 1
        assos_hand = assos[assos[:, 10] == 3]
        for i, asso in enumerate(assos_hand):
            cls = int(asso[11])
            if cls == 0:
                continue
            posx = (asso[1], asso[2], asso[3], asso[4])
            posy = (asso[6], asso[7], asso[8], asso[9])
            idx = pos_to_id[posx]
            if idx not in matched_body_idx:
                continue
            idy = pos_to_id[posy]
            clsx = int(asso[5])
            clsy = int(asso[10])
            if clsx == 0 or clsy == 0:
                continue
            if use_time[cls - 1][idx] < self.clim[cls - 1][clsx - 1] and use_time[cls - 1][idy] < self.clim[cls - 1][clsy - 1]:
                new_assos.append(asso)
                use_time[cls - 1][idx] += 1
                use_time[cls - 1][idy] += 1
        
        next_dt_bbox = []
        for i in range(N):
            bbox = rois[i]
            cls = int(bbox[-1])
            pos = (bbox[1], box[2], box[3], box[4])
            if (use_time[0][i] + use_time[1][i]) < (self.clim[0][cls - 1] + self.clim[1][cls - 1]): # this bbox next match
                next_dt_bbox.append(bbox)
        next_dt_bbox = np.array(next_dt_bbox)

        input['dt_assos'] = np.array(new_assos)
        input['next_dt_bbox'] = next_dt_bbox
        input['use_time'] = use_time
        return input 

class GreedyProcessor(SingleFramePostProcessor):

    def __init__(self, connection_limits):
        self.clim = connection_limits

    def process(self, input):

        rois = input['dt_bboxes']
        assos = input['dt_assos']
        rois, assos = map(to_np_array, [rois, assos])
        
        C = len(self.clim)
        N = rois.shape[0]
        M = assos.shape[0]
        if 'use_time' not in input:
            use_time = [[0 for _ in range(N)] for __ in range(C)]
        else:
            use_time = input['use_time']
        pos_to_id = {}
        new_assos = []
        for i in range(N):
            box = rois[i]
            pos = (box[1], box[2], box[3], box[4])
            pos_to_id[pos] = i

        order = np.argsort(assos[:, 12].reshape(-1))
        order = order[::-1]
        assos = assos[order]
        for i in range(M):
            asso = assos[i]
            cls = int(asso[11])
            if cls == 0:
                continue
            posx = (asso[1], asso[2], asso[3], asso[4])
            posy = (asso[6], asso[7], asso[8], asso[9])
            idx = pos_to_id[posx]
            idy = pos_to_id[posy]
            clsx = int(asso[5])
            clsy = int(asso[10])
            if clsx == 0 or clsy == 0:
                continue
            if use_time[cls - 1][idx] < self.clim[cls - 1][clsx - 1] and use_time[cls - 1][idy] < self.clim[cls - 1][clsy - 1]:
                new_assos.append(asso)
                use_time[cls - 1][idx] += 1
                use_time[cls - 1][idy] += 1
        next_dt_bbox = []
        for i in range(N):
            bbox = rois[i]
            cls = int(bbox[-1])
            pos = (bbox[1], box[2], box[3], box[4])
            if (use_time[0][i] + use_time[1][i]) < (self.clim[0][cls - 1] + self.clim[1][cls - 1]): # this bbox next match
                next_dt_bbox.append(bbox)
        next_dt_bbox = np.array(next_dt_bbox)

        input['dt_assos'] = np.array(new_assos)
        input['next_dt_bbox'] = next_dt_bbox
        input['use_time'] = use_time
        return input 

class GreedyTwiceProcessor(SingleFramePostProcessor):

    def __init__(self, connection_limits):
        self.clim = connection_limits

    def process(self, input):

        rois = input['dt_bboxes']
        assos = input['dt_assos']
        rois, assos = map(to_np_array, [rois, assos])
        pdb.set_trace()
        C = len(self.clim)
        N = rois.shape[0]
        M = assos.shape[0]
        use_time = [[0 for _ in range(N)] for __ in range(C)]
        pos_to_id = {}
        new_assos = []
        for i in range(N):
            box = rois[i]
            pos = (box[1], box[2], box[3], box[4])
            pos_to_id[pos] = i

        order = np.argsort(assos[:, 12].reshape(-1))
        order = order[::-1]
        assos = assos[order]
        #pdb.set_trace()
        for i in range(M):
            asso = assos[i]
            cls = int(asso[11])
            if cls == 0:
                continue
            posx = (asso[1], asso[2], asso[3], asso[4])
            posy = (asso[6], asso[7], asso[8], asso[9])
            idx = pos_to_id[posx]
            idy = pos_to_id[posy]
            clsx = int(asso[5])
            clsy = int(asso[10])
            if clsx == 0 or clsy == 0:
                continue
            if use_time[cls - 1][idx] < self.clim[cls - 1][clsx - 1] and use_time[cls - 1][idy] < self.clim[cls - 1][clsy - 1]:
                new_assos.append(asso)
                use_time[cls - 1][idx] += 1
                use_time[cls - 1][idy] += 1

        input['dt_assos'] = np.array(new_assos)
        return input 

