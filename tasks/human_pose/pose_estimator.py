import os
import json

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from trt_pose import coco
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from torch2trt import TRTModule


class PoseEstimator():

    DEVICE = torch.device('cuda')
    IMG_SIZE = (256, 256)
    MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    @staticmethod
    def get_topology():
        json_file = 'human_pose.json'
        with open(json_file, 'r') as f:
            human_pose = json.load(f)
        topology = coco.coco_category_to_topology(human_pose)
        return topology
    
    @staticmethod
    def get_model():
        weight_file = 'weights/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        model = TRTModule()
        model.load_state_dict(torch.load(weight_file))
        return model

    @staticmethod
    def preproc(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, PoseEstimator.IMG_SIZE)
        img = Image.fromarray(img)
        img = transforms.functional.to_tensor(img).to(PoseEstimator.DEVICE)
        img.sub_(PoseEstimator.MEAN[:, None, None])
        img.div_(PoseEstimator.STD[:, None, None])
        return img[None, ...]

    def __init__(self):
        topology = self.get_topology()
        self._model = self.get_model()
        self._parse_obj = ParseObjects(topology)
        self._draw_obj = DrawObjects(topology)

    def estimate(self, img):
        data = self.preproc(img)
        cmap, paf = self._model(data)
        cmap = cmap.detach().cpu()
        paf = paf.detach().cpu()
        cnts, objs, peaks = self._parse_obj(cmap, paf)
        self._draw_obj(img, cnts, objs, peaks)


if __name__ == '__main__':
    video_file = '/datasets/samples/pedestrian.mp4'

    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(w, h)  # side

    w_gap = int((w - s) / 2)
    h_gap = int((h - s) / 2)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000.0 / fps)

    estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[h_gap:h_gap+s, w_gap:w_gap+s, :]
        estimator.estimate(frame)
        frame = cv2.resize(frame, (s, s))

        cv2.imshow('result', frame)
        if cv2.waitKey(delay_ms) == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
