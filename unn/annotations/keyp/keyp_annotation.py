import json
import copy
import pdb
import numpy as np
import torch
from ..base_annotation import BaseAnnotation


class KeypAnnotation(BaseAnnotation):

    def __init__(self, cfg):
        super(KeypAnnotation, self).__init__('keyp', cfg)
        self.output_names = ['gt_keyps']
        self.num_keypoints = cfg.get('num_keypoints', 17)
        self.keyp_pairs = cfg.get('keyp_pairs', [])
        self.keyp_list = []

    def parse_gt(self, gt_content):
        tmp_keyp_list = []

        keypoints = gt_content.get('keypoints', [])
        for item in keypoints:
            tmp_keyp = []
            for i in range(self.num_keypoints):
                tmp_keyp.append(item[i * 3 : i * 3 + 3])
            tmp_keyp_list.append(tmp_keyp)

        self.keyp_list.append(tmp_keyp_list)

    def generate_anno(self, idx, cfg):
        gt_keypoints = self.keyp_list[idx]
        
        if len(gt_keypoints) == 0:
            gt_keypoints = self._fake_zero_data(1, self.num_keypoints, 3)

        gt_keypoints = torch.as_tensor(gt_keypoints, dtype=torch.float32)
        
        output = {
            'gt_keyps': gt_keypoints
        }

        return output

    def batch_transform(self, annos, cfg):
        output = super().batch_transform(annos, cfg)
        batch = len(output['gt_keyps'])
        for i in range(batch):
            keyp = output['gt_keyps'][i]
            if output['image_info'][5] == 1:
                clone_keyp = copy.deepcopy(keyp)
                for left, right in self.keyp_pairs:
                    keyp[:, left, :] = keyp[:, right, :]
                    keyp[:, right, :] = clone_keyp[:, left, :]
                output['gt_keyps'][i] = keyp

            output['gt_keyps'][i] = torch.as_tensor(output['gt_keyps'][i], dtype=torch.float32)
        return output

    def _fake_zero_data(self, *size):
        return torch.zeros(size)

    def parse_res(self, res_file):
        self.results = {}
        for line in open(res_file):
            res = json.loads(line)
            if not 'keypoints' in res:
                continue
            image_id = res['image_id']
            if not image_id in self.results:
                self.results[image_id] = {'keypoints': []}
            self.results[image_id]['keypoints'].append(res)

    def dump(self, writer, output):
        filename = output['filename']
        image_info = self.tensor2numpy(output['image_info'])
        bboxes = self.tensor2numpy(output.get('dt_bboxes', []))
        if bboxes.shape[0] == 0:
            return
        person_idx = np.where(bboxes[:, 6] == 1)[0]
        bboxes = bboxes[person_idx]
        keyps = self.tensor2numpy(output.get('dt_keyps', []))
        if len(bboxes) == 0:
            return
        dump_results = []
        for b_ix in range(len(image_info)):
            scale_factor = image_info[b_ix][2]
            img_id = filename[b_ix]

            scores = bboxes[:, 5]
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)
            img_bboxes = bboxes[keep_ix]
            if len(keyps) > 0:
                img_keyps = keyps[keep_ix]
            img_bboxes[:, 1: 1 + 4] /= scale_factor
            img_keyps[:, :, :2] /= scale_factor
            N = img_keyps.shape[0]
            for i, bbox in enumerate(img_bboxes):
                keyp = img_keyps[i]
                keyp_list = []
                for j in range(keyp.shape[0]):
                    for k in range(keyp.shape[1]):
                        keyp_list.append(keyp[j][k].tolist())
                res = {
                    'image_id': img_id,
                    'keyp_bbox': bbox[1: 1 + 4].tolist(),
                    'score': float(bbox[5]),
                    'label': int(bbox[6]),
                    'keypoints': keyp_list
                }
                dump_results.append(json.dumps(res, ensure_ascii=False))

        writer.write('\n'.join(dump_results) + '\n')
        writer.flush()
