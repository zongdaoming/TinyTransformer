import copy
import logging

import torch
from .brush_factory import BrushFactory

logger = logging.getLogger('global')


class BaseAnnotation:

    def __init__(self, name, cfg):
        self.brush_init = False
        self.name = name
        self.cfg = copy.deepcopy(cfg)

    def parse_gt(self, gt_content):
        pass

    def generate_anno(self, idx):
        pass

    def parse_res(self, res_file):
        pass

    def dump(self, writer, output):
        pass

    def batch_transform(self, anno, cfg):
        output = {}
        for name in self.output_names:
            batch_anno = [_.get(name, None) for _ in anno]
            output[name] = batch_anno
        return output

    def visualize(self, img, image_id):
        if not self.brush_init:
            self.brush_init = True
            self.brush = BrushFactory.create(self.name, self.cfg['visual'])
        if image_id in self.results:
            self.brush.draw(img, self.results[image_id])
        return img

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x
