# resize ufc101,hmdb51 到128x171,然后 取112x112

import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F
import sys

sys.path.append('..')
from lib.processing_utils import get_file_list, replace_string
import os

worker = torchvision.transforms.Resize((256))
origin_path = "/home/ssd/video_action/ucf101/rawframes"
origin_path_list = get_file_list(origin_path)
origin_path_list.sort()
total_len = len(origin_path_list)
for i in range(len(origin_path_list)):
    file = origin_path_list[total_len-i-1]
    print(i)
    action = file.split('/')[-3]
    video = file.split('/')[-2]
    file_name = file.split('/')[-1]
    save_dir = os.path.join("/home/ssd/video_action/ucf101/rawframes_resize", action, video)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    img = Image.open(file)
    img_resize = worker(img)
    img_resize.save(save_path, quality=90)
    i += 1
    if i > (total_len // 2) + 200:
        sys.exit(0)

    print(i)
