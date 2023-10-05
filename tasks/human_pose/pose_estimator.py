import os
import json

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from trt_pose import coco
from trt_pose import plugins
from torch2trt import TRTModule


class PoseEstimator():

    DEVICE = torch.device('cuda')
    MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    @staticmethod
    def get_topology():
        with open('human_pose.json', 'r') as f:
            metadata = json.load(f)
        topology = coco.coco_category_to_topology(metadata)
        return topology

    def __init__(self, weight_file, input_img_width, input_img_height):
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(weight_file))
        self.input_img_size = (input_img_width, input_img_height)
        self.parser = PoseParser(self.get_topology())

    def preproc(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_img_size)
        img = Image.fromarray(img)
        img = transforms.functional.to_tensor(img).to(PoseEstimator.DEVICE)
        img.sub_(PoseEstimator.MEAN[:, None, None])
        img.div_(PoseEstimator.STD[:, None, None])
        return img[None, ...]

    def parse_pose(self, img):
        data = self.preproc(img)
        cmap, paf = self.model(data)
        cmap = cmap.detach().cpu()
        paf = paf.detach().cpu()
        poses = self.parser(img, cmap, paf)
        return poses


class PoseEstimatorResNet(PoseEstimator):
    def __init__(self):
        weight_file = 'weights/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        img_width, img_height = (224, 224)
        super().__init__(weight_file, img_width, img_height)


class PoseEstimatorDenseNet(PoseEstimator):
    def __init__(self):
        weight_file = 'weights/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        img_width, img_height = (256, 256)
        super().__init__(weight_file, img_width, img_height)


class PoseParser():
    def __init__(self,
                 topology,
                 cmap_threshold=0.05,
                 link_threshold=0.05,
                 cmap_window=5,
                 line_integral_samples=7,
                 max_num_parts=100,
                 max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects

    def __call__(self, img, cmap, paf):
        num_objs, objs, norm_peaks = self._parse_object(cmap, paf)
        kpts = self._parse_keypoints(img, num_objs, objs, norm_peaks)
        return kpts

    def _parse_object(self, cmap, paf):
        num_peak, peaks = plugins.find_peaks(
            cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        norm_peaks = plugins.refine_peaks(
            num_peak, peaks, cmap, self.cmap_window)
        score_graph = plugins.paf_score_graph(
            paf, self.topology, num_peak, norm_peaks, self.line_integral_samples)
        connections = plugins.assignment(
            score_graph, self.topology, num_peak, self.link_threshold)
        num_objs, objs = plugins.connect_parts(
            connections, self.topology, num_peak, self.max_num_objects)
        return num_objs, objs, norm_peaks

    def _parse_keypoints(self, img, num_objs, objs, norm_peaks):
        img_height, img_width = img.shape[:2]

        num_objs = num_objs.squeeze()
        objs = objs.squeeze()
        norm_peaks = norm_peaks.squeeze()

        num_joints = objs[0].size()[0]  # 18
        poses = torch.zeros(size=(num_objs, num_joints, 2), dtype=torch.int16)
        for i, obj in enumerate(objs):
            for j in range(num_joints):
                k = int(obj[j])
                if k >= 0:
                    joint_y, joint_x = norm_peaks[j][k]
                    poses[i, j, 0] = round(float(joint_x) * img_width)
                    poses[i, j, 1] = round(float(joint_y) * img_height)

        return poses
