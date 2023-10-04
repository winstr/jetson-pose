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


class ParseObjects(object):
    def __init__(self, topology, cmap_threshold=0.05, link_threshold=0.05, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects

    def __call__(self, cmap, paf):
        peak_counts, peaks = plugins.find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        normalized_peaks = plugins.refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        score_graph = plugins.paf_score_graph(paf, self.topology, peak_counts, normalized_peaks, self.line_integral_samples)
        connections = plugins.assignment(score_graph, self.topology, peak_counts, self.link_threshold)
        object_counts, objects = plugins.connect_parts(connections, self.topology, peak_counts, self.max_num_objects)
        return object_counts, objects, normalized_peaks


class DrawObjects(object):
    def __init__(self, topology):
        self.topology = topology

    def __call__(self, img, num_objs, objs, norm_peaks):
        img_width, img_height = img.shape[:2][::-1]
        num_objs = int(num_objs[0])
        objs = objs.squeeze()
        norm_peaks = norm_peaks.squeeze()
        num_kpts = objs[0].size()[0]
        kptss = torch.zeros(size=(num_objs, num_kpts, 2), dtype=torch.int16)
        for i, obj in enumerate(objs):
            for j in range(num_kpts):
                k = int(obj[j])
                if k >= 0:
                    y, x = norm_peaks[j][k]
                    kptss[i, j, 0] = round(float(x) * img_width)
                    kptss[i, j, 1] = round(float(y) * img_height)
        return kptss


"""
class DrawObjects(object):
    def __init__(self, topology):
        self.topology = topology

    def __call__(self, image, object_counts, objects, normalized_peaks):
        height, width = image.shape[:2]

        count = int(object_counts[0])
        K = self.topology.shape[0]
        
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)

            for k in range(K):
                c_a = self.topology[k][2]
                c_b = self.topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
"""