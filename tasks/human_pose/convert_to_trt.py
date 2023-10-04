import os
import json

import torch

from trt_pose import models
from torch2trt import torch2trt


def convert_to_trt(pth_file):
    if not os.path.isfile(pth_file):
        raise FileNotFoundError(pth_file)
    
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    pth_name = os.path.basename(pth_file)
    if pth_name == 'resnet18_baseline_att_224x224_A_epoch_249.pth':
        model = models.resnet18_baseline_att(num_parts, 2 * num_links)
        width, height = 224, 224
    elif pth_name == 'densenet121_baseline_att_256x256_B_epoch_160.pth':
        model = models.densenet121_baseline_att(num_parts, 2 * num_links)
        width, height = 256, 256
    else:
        raise ValueError(f'Unsupported: {pth_name}')

    model.cuda().eval()
    model.load_state_dict(torch.load(pth_file))
    
    data = torch.zeros((1, 3, height, width)).cuda()
    trt_model = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    trt_pth_file = os.path.splitext(pth_file)[0] + "_trt.pth"

    torch.save(trt_model.state_dict(), trt_pth_file)


if __name__ == '__main__':
    pth_files = [
        'weights/resnet18_baseline_att_224x224_A_epoch_249.pth',
        'weights/densenet121_baseline_att_256x256_B_epoch_160.pth',]
    convert_to_trt(pth_files[1])