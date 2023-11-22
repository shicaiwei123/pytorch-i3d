from torch.utils.data import Dataset

from lib.processing_utils import read_txt, replace_string
import glob
from PIL import Image
import numpy as np
from utils import util
import datetime
import torch
import os
import pickle
import time
import torchvision.datasets as td


class KINECTS400(Dataset):
    def __init__(self, data_root, split_path, clip_len, mode, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-HMDB51 数据集的路径
        :param split_path: path for split txt
        :param clip_len: 每个动作取多少帧用于训练
        :mode : rgb or depth
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(KINECTS400, self).__init__()

        self.test_data = []
        self.train_data = []

        self.clip_len = clip_len
        frames_path_len_min = 10000
        self.mode = mode

        txt_name = split_path.split('/')[-1]
        print(txt_name)
        if 'train' in txt_name.split('_')[1]:
            self.train = True
        elif 'val' in txt_name.split('_')[1]:
            self.train = False
        else:
            print('error file')

        # split_line = read_txt(split_path)
        # for line in split_line:
        #     related_path = line.split(' ')[0]
        #     related_path = related_path.split('.')[0]
        #     label = int(line.split(' ')[1])
        #     video_path = os.path.join(data_root, related_path)
        #     print(video_path)
        #
        #     img_list = os.listdir(video_path)
        #     img_len = len(img_list)
        #     # print(img_len,img_len/3)
        #     # print(video_path)
        #     if np.mod(img_len, 3) != 1:
        #         print(video_path)
        #         with open('img_len_error.txt', 'a+') as f:
        #             f.write(video_path)
        #         continue
        #     select_num = img_len // 3
        #     rgb_list = []
        #     flow_x_list = []
        #     flow_y_list = []
        #     for i in range(select_num):
        #         i = str(i)
        #         i = i.zfill(5)
        #         rgb_list.append(os.path.join(video_path, "img_" + i + ".jpg"))
        #         flow_x_list.append(os.path.join(video_path, "flow_x_" + i + ".jpg"))
        #         flow_y_list.append(os.path.join(video_path, "flow_y_" + i + ".jpg"))

        # flow_x_list = img_list[0:select_num]
        # flow_y_list = img_list[select_num:select_num * 2]
        # rgb_list = img_list[select_num * 2:select_num * 3 - 1]

        # if len(flow_x_list) < clip_len * 2:
        #     continue
        # if self.train:
        #     self.train_data.append(
        #         {"frames": rgb_list, "flow_x": flow_x_list, "flow_y": flow_y_list, "label": label})
        # else:
        #     self.test_data.append(
        #         {"frames": rgb_list, "flow_x": flow_x_list, "flow_y": flow_y_list, "label": label})

        # if self.train:
        #
        #     with open("train_data_dict.txt", 'a+') as f:
        #         for fp in self.train_data:
        #             f.write(str(fp))
        #             f.write('\n')
        # else:
        #     with open("test_data_dict.txt", 'a+') as f:
        #         for fp in self.test_data:
        #             f.write(str(fp))
        #             f.write('\n')

        if self.train:
            self.train_data = read_txt("../src/train_data_dict.txt")
            for i in range(len(self.train_data)):
                # print(i)
                self.train_data[i] = eval(self.train_data[i])
                # if i > 2500:
                #     break
                # if int(self.train_data[i]['label']) > 196:
                #     self.train_data[i]['label'] = self.train_data[i]['label'] - 1
                # if i > 1000:
                #     self.train_data=self.train_data[0:1000]
                #     break
                # print(i)
                print(self.train_data[i]['label'])
            # self.train_data = self.train_data[0:2000]


        else:
            self.test_data = read_txt("../src/test_data_dict.txt")
            for i in range(len(self.test_data)):
                # print(i)
                # if i > 250:
                #     break
                self.test_data[i] = eval(self.test_data[i])
                # if i > 1000:
                #     self.train_data=self.train_data[0:1000]
                #     break
            # self.train_data = self.train_data[0:2000]

        if self.train:
            self.data = self.train_data
        else:
            self.data = self.test_data

        if self.train:
            self.clip_transform = util.clip_transform_kinects('train', clip_len)
        else:
            self.clip_transform = util.clip_transform_kinects('val', clip_len)

    def loader(self, image_filename):
        read_List = []
        for path in image_filename:
            read_List.append(Image.open(path))

        return read_List

    def sample_rgb(self, entray):
        imgs = entray['frames']
        if len(imgs) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(imgs) - self.clip_len)
                imgs = imgs[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(imgs) // 2 - self.clip_len // 2
                imgs = imgs[offset:offset + self.clip_len]
            assert len(imgs) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        rgb_imgs = [self.loader(img) for img in imgs]
        return rgb_imgs

    def sample_flow(self, entray):
        flow_x = entray['flow_x']
        flow_y = entray['flow_y']

        # print(len(flow_x))
        if len(flow_x) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(flow_x) - self.clip_len)
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(flow_x) // 2 - self.clip_len // 2
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            flow_x = flow_x
            flow_y = flow_y
        aa = time.time()
        # print("flowx_len", len(flow_x))
        # print(flow_x)
        # flow_y_imgs = []
        # flow_x_imgs=[]
        # flow_y_imgs= [self.loader(img) for img in flow_y]
        img_read = list(map(self.loader, [flow_x, flow_y]))
        flow_x_imgs, flow_y_imgs = img_read[0], img_read[1]

        return flow_x_imgs, flow_y_imgs


        # flow_x_imgs = list(map(self.loader,flow_x))
        # bb = time.time()
        # # print("flow_x", bb - aa)
        # flow_y_imgs = list(map(self.loader,flow_y))

    def sample_all(self, entry):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img = entry['frames']
        flow_x = entry['flow_x']
        flow_y = entry['flow_y']

        # print(len(flow_x))

        if len(flow_x) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(flow_x) - self.clip_len)
                rgb_img = rgb_img[offset:offset + self.clip_len]
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(flow_x) // 2 - self.clip_len // 2
                # print(offset, offset + self.clip_len * 2, len(flow_x))
                rgb_img = rgb_img[offset:offset + self.clip_len]
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            assert len(rgb_img) == self.clip_len, 'frame selection error!'
        else:
            # print(len(flow_x), len(flow_x), len(flow_x))
            raise RuntimeError("len(imgs) > self.clip_len")

        rgb_imgs = [self.loader(img) for img in rgb_img]
        flow_x_imgs = [self.loader(img) for img in flow_x]
        flow_y_imgs = [self.loader(img) for img in flow_y]

        return rgb_imgs, flow_x_imgs, flow_y_imgs

    def __getitem__(self, index):
        entry = self.data[index]
        if self.mode == 'rgb':
            rgb_frames = self.sample_rgb(entry)
            rgb_frames = self.clip_transform(rgb_frames)  # (T, 3, 224, 224)
            rgb_frames = rgb_frames.permute(1, 0, 2, 3)  # (3, T, 224, 224)
            b = datetime.datetime.now()
            # print((b-c).total_seconds())
            # print(entry['label'])
            instance = {'frames': rgb_frames, 'label': entry['label']}
        elif self.mode == 'flow':
            # print(entry)

            aa = time.time()
            flow_x, flow_y = self.sample_flow(entry)
            bb = time.time()
            # print("sample_all", bb - aa)
            input = {"flow_x": flow_x, "flow_y": flow_y}
            output = self.clip_transform(input)
            cc = time.time()
            # print("transform", cc - bb)
            flow_x, flow_y = output["flow_x"], output["flow_y"]
            zero_z = torch.zeros_like(flow_x)
            flow_xy = torch.cat((flow_x, flow_y), dim=1)
            flow_xy = flow_xy.permute(1, 0, 2, 3)
            instance = {'flow': flow_xy, 'label': entry['label']}
        elif self.mode == 'all':
            aa = time.time()
            rgb_frames, flow_x, flow_y = self.sample_all(entry)
            bb = time.time()
            print("sample_all", bb - aa)
            input = {"rgb": rgb_frames, "flow_x": flow_x, "flow_y": flow_y}
            output = self.clip_transform(input)
            cc = time.time()
            # print("transform", cc - bb)
            rgb_frames, flow_x, flow_y = output["rgb"], output["flow_x"], output["flow_y"]
            flow_xy = torch.cat((flow_x, flow_y), dim=1)
            flow_xy = flow_xy.permute(1, 0, 2, 3)
            rgb_frames = rgb_frames.permute(1, 0, 2, 3)
            instance = {'frames': rgb_frames, 'flow': flow_xy, 'label': entry['label']}

        else:
            raise RuntimeError("self.mode=='rgb'")

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)
