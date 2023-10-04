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
    MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    @staticmethod
    def get_topology():
        json_file = 'human_pose.json'
        with open(json_file, 'r') as f:
            human_pose = json.load(f)
        topology = coco.coco_category_to_topology(human_pose)
        return topology

    def __init__(self, weight_file, img_width, img_height):
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(weight_file)
        self.weight_file = weight_file
        self.img_size = (img_width, img_height)

        topology = self.get_topology()
        self.model = self._load_model()
        self.parse_obj = ParseObjects(topology)
        self.draw_obj = DrawObjects(topology)

    def _load_model(self):
        model = TRTModule()
        model.load_state_dict(torch.load(self.weight_file))
        return model
    
    def _preprc(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = Image.fromarray(img)
        img = transforms.functional.to_tensor(img).to(PoseEstimator.DEVICE)
        img.sub_(PoseEstimator.MEAN[:, None, None])
        img.div_(PoseEstimator.STD[:, None, None])
        return img[None, ...]

    def estimate(self, img):
        data = self._preprc(img)
        cmap, paf = self.model(data)
        cmap = cmap.detach().cpu()
        paf = paf.detach().cpu()
        cnts, objs, peaks = self.parse_obj(cmap, paf)
        self.draw_obj(img, cnts, objs, peaks)


class ResNetPoseEstimator(PoseEstimator):
    def __init__(self):
        weight_file = 'weights/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        img_width, img_height = (224, 224)
        super().__init__(weight_file, img_width, img_height)


class DenseNetPoseEstimator(PoseEstimator):
    def __init__(self):
        weight_file = 'weights/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        img_width, img_height = (256, 256)
        super().__init__(weight_file, img_width, img_height)
