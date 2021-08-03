import cv2

class HumanPartsBrush:

    def __init__(self, vis_score=False, score_thresh=0.0, line_width=4, line_color=(255, 255, 255)):
        self.vis_score = vis_score
        self.score_thresh = score_thresh
        self.line_width = line_width
        self.line_color = line_color

    def draw(self, img, result):
        assos = result['assos']
        for item in assos:
            score = item['score']
            if score < self.score_thresh:
            #if score > 0.5:
                continue
            bbox1 = item['bbox1']
            cx1 = (bbox1[0] + bbox1[2]) / 2.0
            cy1 = (bbox1[1] + bbox1[3]) / 2.0
            bbox2 = item['bbox2']
            cx2 = (bbox2[0] + bbox2[2]) / 2.0
            cy2 = (bbox2[1] + bbox2[3]) / 2.0
            cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
            cv2.line(img, (cx1, cy1), (cx2, cy2), self.line_color, self.line_width)
            if self.vis_score:
                score = item['score']
                x = bbox2[0] - 10
                y = (bbox2[1] + bbox2[3]) / 2.0 - 10
                #cv2.putText(img, str(score), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                cv2.putText(img, '{:.4f}'.format(score), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

       
        return img

