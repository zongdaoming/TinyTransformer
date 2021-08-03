import json
import cv2
# import mc

import os
import logging
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pdb

from ..base_annotation import BaseAnnotation

cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger('global')


class ImageAnnotation(BaseAnnotation):

    def __init__(self, cfg):
        super(ImageAnnotation, self).__init__('asso', cfg)
        self.output_names = ['image', 'filename', 'image_info', 'gt_labels']
        self.filename_list = []
        self.aspect_ratios = []
        self.image_labels = []
        self.root_dir = cfg.get('image_dir', None)
        self.initialized = False
        self.aspect_warning = False
        self.memcached = cfg.get('memcached', False)
        if 'pixel_mean' in cfg and 'pixel_std' in cfg:
            self.normalize_fn = transforms.Normalize(mean=cfg['pixel_mean'], std=cfg['pixel_std'])
        else:
            self.normalize_fn = None

    def parse_gt(self, gt_content):
        self.filename_list.append(gt_content['filename'])
        if 'image_height' in gt_content and 'image_width' in gt_content:
            image_height = gt_content['image_height']
            image_width = gt_content['image_width']
            self.aspect_ratios.append(image_height / image_width)
        else:
            if not self.aspect_warning:
                logger.warning('image_size is not provided, which may lead to aspect grouping policy useless')
                self.aspect_warning = True
            self.aspect_ratios.append(1)
        if 'label' in gt_content:
            self.image_labels.append(gt_content['label'])
        else:
            self.image_labels.append(0)

    def generate_anno(self, idx, cfg):

        origin_filename = self.filename_list[idx]
        filename = os.path.join(self.root_dir, self.filename_list[idx])
        if not os.path.exists(filename) and not cfg.get('ceph', False):
            logger.error('filepath not exist: ' + filename)

        # if self.memcached:
        #     self._init_memcached()
        #     value = mc.pyvector()
        #     assert len(filename) < 250, 'memcached requires length of path < 250'
        #     self.mclient.Get(filename, value)
        #     value_buf = mc.ConvertBuffer(value)
        #     img_array = np.frombuffer(value_buf, np.uint8)
        #     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # elif cfg.get('ceph', False):
        #     import ceph
        #     s3client = ceph.S3Client()
        #     try:
        #         value_buf = s3client.Get(self.filename_list[idx])
        #     except Exception as e:
        #         logger.error(e)
        #         logger.error('filepath not exist: ' + self.filename_list[idx])
        #     img_array = np.frombuffer(value_buf, np.uint8)
        #     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # else:


        img = cv2.imread(filename, cv2.IMREAD_COLOR)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_h, image_w = img.shape[:2]

        output = {
            'image': img,
            'filename': origin_filename,
            'image_info': np.array([image_h, image_w, 1.0, image_h, image_w, False]),
            'gt_labels': np.array(self.image_labels[idx])
        }
        return output

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self.initialized = True

    def batch_transform(self, annos, cfg):
        output = super().batch_transform(annos, cfg)
        images = output['image']
        for i in range(len(images)):
            images[i] = transforms.functional.to_tensor(images[i])
            if self.normalize_fn is not None:
                images[i] = self.normalize_fn(images[i])
            output['image_info'][i] = torch.as_tensor(output['image_info'][i], dtype=torch.float32)
            output['gt_labels'][i] = torch.as_tensor(output['gt_labels'][i], dtype=torch.float32)
        alignment = cfg.get('alignment', 1)
        pad_value = cfg.get('pad_value', 0)
        max_img_h = max([_.size(-2) for _ in images])
        max_img_w = max([_.size(-1) for _ in images])
        target_h = int(np.ceil(max_img_h / alignment) * alignment)
        target_w = int(np.ceil(max_img_w / alignment) * alignment)
        padded_images = []
        for image in images:
            assert image.dim() == 3
            src_h, src_w = image.size()[-2:]
            pad_size = (0, target_w - src_w, 0, target_h - src_h)
            padded_images.append(F.pad(image, pad_size, 'constant', pad_value).data)
        output['image'] = torch.stack(padded_images)
        output['gt_labels'] = torch.stack(output['gt_labels'])
        return output

    def dump(self, writer, output):
        if 'dt_labels' in output:
            filename = output['filename']
            image_info = self.tensor2numpy(output['image_info'])
            labels = self.tensor2numpy(output['dt_labels'])
            dump_results = []
            for b_ix in range(len(image_info)):
                img_id = filename[b_ix]
                label = int(labels[b_ix][0])
                score = float(labels[b_ix][1])
                res = {
                    'image_id': img_id,
                    'label': label,
                    'score': score
                }
                dump_results.append(json.dumps(res, ensure_ascii=False))

            writer.write('\n'.join(dump_results) + '\n')
            writer.flush()
