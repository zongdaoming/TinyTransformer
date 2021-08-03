import cv2
import random

class GeneralBrush:

    def __init__(self, vis_score=False, vis_label=False, vis_track=False, label_name=None, rec_width=4, rec_color=None, text_color=None, font_scale=2, use_track=False):
        self.vis_score = vis_score
        self.rec_width = rec_width
        self.rec_color = rec_color
        self.text_color = text_color
        self.use_track = use_track
        self.vis_label = vis_label
        self.vis_track = vis_track
        self.label_name = label_name
        self.font_scale = font_scale
        if self.use_track:
            self.mapped_color = {}

    def draw(self, img, result):
        bboxes = result['bboxes']
        for item in bboxes:
            bbox = item['bbox']
            cls_idx = item['label'] - 1
            if self.use_track:
                track_id = item['track_id']
                if track_id in self.mapped_color:
                    color = self.mapped_color[track_id]
                else:
                    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    color = (r, g, b)
                    self.mapped_color[track_id] = color

            elif self.rec_color is None:
                r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                color = (r, g, b)
            else:
                if isinstance(self.rec_color[0], list):
                    color = self.rec_color[cls_idx]
                else:
                    color = self.rec_color
            if self.text_color is None:
                text_color = (0, 0, 0)
            else:
                if isinstance(self.text_color[0], list):
                    text_color = self.text_color[cls_idx]
                else:
                    text_color = self.text_color
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.rec_width)
            put_text = False
            text = ''
            if self.vis_label:
                x = int(bbox[0] - 10)
                y = int((bbox[1] + bbox[3]) / 2.0 - 10)
                if self.label_name is not None:
                    label = self.label_name[item['label'] - 1]
                    text = '%s' % label
                else:
                    label = item['label']
                    text = '%d' % label
                put_text = True
            if self.vis_track:
                text = text + ' id: %d' % item['track_id']
                put_text = True
            if self.vis_score:
                score = item['score']
                x = int(bbox[0] - 10)
                y = int((bbox[1] + bbox[3]) / 2.0 - 10)
                text = text + '   %.2f' % score
                put_text = True
            if put_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                ((txt_w, txt_h), _) = cv2.getTextSize(text, font, self.font_scale, 1)
                back_tl = (int(bbox[0]), int(bbox[1]) - int(1.3 * txt_h))
                back_br = (int(bbox[0]) + txt_w, int(bbox[1]))
                cv2.rectangle(img, back_tl, back_br, color, -1)
                txt_tl = int(bbox[0]), int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(img, text, txt_tl, font, self.font_scale, text_color, lineType=cv2.LINE_AA, thickness=3)
                

        return img

