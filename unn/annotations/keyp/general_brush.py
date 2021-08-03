import cv2
import random

class GeneralBrush:

    def __init__(self, vis_score=False, score_thresh=0.0, line_width=4, line_color=None, skeleton=None):
        self.vis_score = vis_score
        self.score_thresh = score_thresh
        self.line_width = line_width
        self.skeleton = skeleton
        self.line_color = []
        if line_color is None:
            for i in range(len(self.skeleton)):
                r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                self.line_color.append((r, g, b))
        else:
            for i in range(len(self.skeleton)):
                self.line_color.append(line_color)

    def draw(self, img, result):
        keyps = result['keypoints']
        for item in keyps:
            cx = []
            cy = []
            score = []
            length = len(item['keypoints'])
            for i in range(length // 3):
                cx.append(int(item['keypoints'][i * 3]))
                cy.append(int(item['keypoints'][i * 3 + 1]))
                score.append(item['keypoints'][i * 3 + 2])
            for i, line in enumerate(self.skeleton):
                cx1 = cx[line[0]]
                cy1 = cy[line[0]]
                cx2 = cx[line[1]]
                cy2 = cy[line[1]]
                if score[line[0]] < self.score_thresh: continue
                if score[line[1]] < self.score_thresh: continue
                cv2.line(img, (cx1, cy1), (cx2, cy2), self.line_color[i], self.line_width)
         
        return img

