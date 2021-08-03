import json
import copy

import numpy as np
import torch
from ..base_annotation import BaseAnnotation
from .brush_factory import BrushFactory

class BboxAnnotation(BaseAnnotation):

    def __init__(self, cfg):
        super(BboxAnnotation, self).__init__('bbox', cfg)
        self.output_names = ['gt_bboxes', 'gt_ignores', 'gt_attris']
        self.num_classes = cfg.get('num_classes', 2)
        self.num_attributes = cfg.get('num_attributes', 1)
        self.bbox_list = []
        self.attri_list = []
        self.ignore_list = []
        self.base_attribute = [0 for _ in range(self.num_attributes)]
        self.len_attribute = self.num_attributes

    def parse_gt(self, gt_content):
        tmp_bbox_list = []
        tmp_ignore_list = []
        tmp_attri_list = []

        instances = gt_content.get('instances', [])
        for item in instances:
            is_ignored = item.get('is_ignored', False)
            if item.get('label', 1) >= self.num_classes:
                continue
            bbox = item['bbox'] + [item.get('label', 1)]
            attri = copy.deepcopy(self.base_attribute)
            for idx in item.get('attribute', []):
                attri[idx - 1] = 1
            if is_ignored:
                tmp_ignore_list.append(bbox[:4])
            else:
                tmp_bbox_list.append(bbox)
                tmp_attri_list.append(attri)

        self.bbox_list.append(tmp_bbox_list)
        self.ignore_list.append(tmp_ignore_list)
        self.attri_list.append(tmp_attri_list)

    def generate_anno(self, idx, cfg):
        gt_bboxes = self.bbox_list[idx]
        ig_bboxes = self.ignore_list[idx]
        gt_attris = self.attri_list[idx]
        
        if len(gt_bboxes) == 0:
            gt_bboxes = self._fake_zero_data(1, 5)
        if len(ig_bboxes) == 0:
            ig_bboxes = self._fake_zero_data(1, 4)
        if len(gt_attris) == 0:
            gt_attris = self._fake_zero_data(1, self.len_attribute)

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        ig_bboxes = np.array(ig_bboxes, dtype=np.float32)
        gt_attris = np.array(gt_attris, dtype=np.float32)
        
        output = {
            'gt_bboxes': gt_bboxes,
            'gt_ignores': ig_bboxes,
            'gt_attris': gt_attris
        }

        return output

    def batch_transform(self, annos, cfg):
        output = super().batch_transform(annos, cfg)
        for i in range(len(output['gt_bboxes'])):
            output['gt_bboxes'][i] = torch.as_tensor(output['gt_bboxes'][i], dtype=torch.float32)
            output['gt_ignores'][i] = torch.as_tensor(output['gt_ignores'][i], dtype=torch.float32)
            output['gt_attris'][i] = torch.as_tensor(output['gt_attris'][i], dtype=torch.float32)
        return output

    def _fake_zero_data(self, *size):
        return np.zeros(size)

    def parse_res(self, res_file):
        self.results = {}
        for line in open(res_file):
            res = json.loads(line)
            if not 'bbox' in res:
                continue
            image_id = res['image_id']
            if not image_id in self.results:
                self.results[image_id] = {'bboxes': []}
            self.results[image_id]['bboxes'].append(res)

    def dump(self, writer, output):
        filename = output['filename']
        image_info = self.tensor2numpy(output['image_info'])
        bboxes = self.tensor2numpy(output.get('dt_bboxes', []))
        attris = self.tensor2numpy(output.get('dt_attris', []))
        if len(bboxes) == 0:
            return
        dump_results = []
        track_id = output['track_id'] if 'track_id' in output else None
        for b_ix in range(len(image_info)):
            scale_factor = image_info[b_ix][2]
            img_id = filename[b_ix]

            scores = bboxes[:, 5]
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)
            img_bboxes = bboxes[keep_ix]
            if len(attris) > 0:
                img_attris = attris[keep_ix]
            if track_id is not None:
                img_track_id = track_id[keep_ix]
            else:
                img_track_id = None
            img_bboxes[:, 1: 1 + 4] /= scale_factor
            
            for i, bbox in enumerate(img_bboxes):
                if len(attris) > 0:
                    attri = img_attris[i].tolist()
                else:
                    attri = []
                res = {
                    'image_id': img_id,
                    'bbox': bbox[1: 1 + 4].tolist(),
                    'score': float(bbox[5]),
                    'label': int(bbox[6]),
                    'attribute': attri
                }
                if img_track_id is not None:
                    res['track_id'] = int(img_track_id[i])
                dump_results.append(json.dumps(res, ensure_ascii=False))

        writer.write('\n'.join(dump_results) + '\n')
        writer.flush()
