import json
import logging
import torch
import numpy as np

from ..base_annotation import BaseAnnotation
from .brush_factory import BrushFactory
from unn.models.heads.utils.pair_helper import pair_nms

logger = logging.getLogger('global')


class AssoAnnotation(BaseAnnotation):

    def __init__(self, cfg):
        super(AssoAnnotation, self).__init__('asso', cfg)
        self.output_names = ['gt_assos', 'global_assos']
        self.num_classes = cfg.get('num_classes', 2)
        self.compute_bbox = cfg.get('compute_bbox', False)
        self.asso_list = []
        self.asso_flag_list = []

    def parse_gt(self, gt_content):
        tmp_asso_list = []

        bbox_cnt = 0
        bbox_idx = {}
        for item in gt_content.get('instances', []):
            bbox_idx[self.bbox2tuple(item['bbox'])] = bbox_cnt
            bbox_cnt += 1

        for item in gt_content.get('associations', []):
            idx1 = self.bbox2id(bbox_idx, item['bbox1'])
            idx2 = self.bbox2id(bbox_idx, item['bbox2'])
            asso = [idx1, idx2, item['label']] + item['bbox1'] + \
            [item['label1']] + item['bbox2'] + [item['label2']] + [1.5, ]
            tmp_asso_list.append(asso)
        if len(tmp_asso_list) == 0:
            self.asso_flag_list.append(0)
        else:
            self.asso_flag_list.append(1)
        self.asso_list.append(tmp_asso_list)

    def bbox2tuple(self, data):
        return (data[0], data[1], data[2], data[3])

    def bbox2id(self, bbox_idx, bbox):
        if not self.compute_bbox:
            assert (self.bbox2tuple(bbox) in bbox_idx)
            return bbox_idx[self.bbox2tuple(bbox)]
        else:
            max_iou = 0.5
            max_idx = -1
            area1 = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            for item in bbox_idx.keys():
                inter_w = max(min(bbox[2], item[2]) - max(bbox[0], item[0]) + 1, 0)
                inter_h = max(min(bbox[3], item[3]) - max(bbox[1], item[1]) + 1, 0)
                inter_area = inter_w * inter_h
                area2 = (item[2] - item[0] + 1) * (item[3] - item[1] + 1)
                iou = float(inter_area) / float(max(area1 + area2 - inter_area, 1))
                if iou > max_iou:
                    max_iou = iou
                    max_idx = bbox_idx[item]
            assert max_idx > -1
            return max_idx

    def generate_anno(self, idx, cfg):
        gt_assos = self.asso_list[idx]
        gt_assos_flag = self.asso_flag_list[idx]

        if len(gt_assos) == 0:
            gt_assos = self._fake_zero_data(1, 13)

        gt_assos = np.array(gt_assos)
        gt_assos_flag = np.array(gt_assos_flag)

        output = {
            'gt_assos': gt_assos,
            'gt_assos_flag': gt_assos_flag
        }
        return output

    def batch_transform(self, annos, cfg):
        output = super().batch_transform(annos, cfg)
        for i in range(len(output['gt_assos'])):
            output['gt_assos'][i] = torch.as_tensor(output['gt_assos'][i], dtype=torch.float32)
        return output

    def _fake_zero_data(self, *size):
        return np.zeros(size)

    def parse_res(self, res_file):
        self.results = {}
        for line in open(res_file):
            res = json.loads(line)
            if not 'asso' in res:
                continue
            image_id = res['image_id']
            bbox1 = res['asso'][0:4]
            bbox2 = res['asso'][4:8]
            score = res['score']
            label = res['label']
            label1 = res['label1']
            label2 = res['label2']
            if not image_id in self.results:
                self.results[image_id] = {'assos': []}
            self.results[image_id]['assos'].append(
                {'bbox1': bbox1, 'bbox2': bbox2, 'score': score, 'label': label, 'label1': label1, 'label2': label2,
                 'image_id': image_id})

    def post_processor(self, res_file):
        dump_results = []
        N = len(self.results.keys())
        for i, image_id in enumerate(self.results.keys()):
            assos = self.results[image_id]['assos']
            batch_assos = []
            for asso in assos:
                batch_assos.append(
                    [0] + asso['bbox1'] + [asso['label1']] + asso['bbox2'] + [asso['label2']] + [asso['label']] + [
                        asso['score']])
            batch_assos = np.array(batch_assos)
            # batch_assos = pair_nms(batch_assos, self.cfg)
            for asso in batch_assos:
                if int(asso[11]) == 0: continue
                res = {
                    'image_id': image_id,
                    'asso': asso[1: 5].tolist() + asso[6: 10].tolist(),
                    'score': float(asso[12]),
                    'label': int(asso[11]),
                    'label1': int(asso[5]),
                    'label2': int(asso[10])
                }
                dump_results.append(json.dumps(res, ensure_ascii=False))
            if i % 100 == 0:
                logger.info('Post Process Progress:[{:.2f}%][{}/{}]'.format(float(i + 1) / float(N) * 100.0, i + 1, N))

        writer = open(res_file, 'w')
        writer.write('\n'.join(dump_results) + '\n')
        writer.flush()

    def dump(self, writer, output):
        filename = output['filename']
        image_info = self.tensor2numpy(output['image_info'])
        assos = self.tensor2numpy(output.get('dt_assos', []))
        if len(assos) == 0:
            return
        dump_results = []
        for b_ix in range(len(image_info)):
            scale_factor = image_info[b_ix][2]
            img_id = filename[b_ix]

            scores = assos[:, 12]
            keep_ix = np.where(assos[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)

            img_assos = assos[keep_ix]
            img_assos[:, 1: 5] /= scale_factor
            img_assos[:, 6: 10] /= scale_factor
            for asso in img_assos:
                if int(asso[11]) == 0: continue
                res = {
                    'image_id': img_id,
                    'asso': asso[1: 5].tolist() + asso[6: 10].tolist(),
                    'score': float(asso[12]),
                    'label': int(asso[11]),
                    'label1': int(asso[5]),
                    'label2': int(asso[10])
                }
                dump_results.append(json.dumps(res, ensure_ascii=False))
        writer.write('\n'.join(dump_results) + '\n')
        writer.flush()
