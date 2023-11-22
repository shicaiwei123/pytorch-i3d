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


class NTUD120_CS(Dataset):
    def __init__(self, data_root, clip_len, mode, train=True, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-UCLA 数据集的路径
        :param split_num: 哪一个subject 用来测试
        :param clip_len: 每个动作取多少帧用于训练
        :mode : rgb or depth
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(NTUD120_CS, self).__init__()
        self.train_data = []
        self.test_data = []
        self.clip_len = clip_len
        self.train_subject_list = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 3845, 46, 47,
                                   49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89,
                                   91, 92, 93, 94, 95, 97, 98, 100, 103]
        frames_path_len_min = 10000
        self.mode = mode

        # 根据mode确定索引路径
        if mode == 'rgb':
            self.data_root = os.path.join(data_root, 'rawframes')
            self.data_root_120 = os.path.join(data_root, 'raw/nturgb+d_rgb')
        elif mode == 'depth':
            self.data_root = os.path.join(data_root, 'Depth')
            self.data_root_120 = os.path.join(data_root, 'ntud_depth_120/ntud_depth_120')


        elif mode == 'all':
            self.rgb_data_root = os.path.join(data_root, 'rawframes')
            self.depth_data_root = os.path.join(data_root, 'Depth')

        else:
            raise ValueError('mode should be rgb or depth')

        self.loader = lambda fl: Image.open(fl)

        sample_list = os.listdir(self.data_root)
        # sample_list.sort(reverse=True)
        for sample in sample_list:
            # print(sample)
            sample_name = sample.split('.')[0]
            subject_num = int(sample_name[9:12])
            sample_label = int(sample_name[17:20]) - 1

            if subject_num in self.train_subject_list:
                sample_path = os.path.join(self.data_root, sample)
                frames_path = glob.glob('%s/*' % (sample_path))

                if len(frames_path) < 10:
                    frames_path_len_min = len(frames_path)

                frames_path.sort()

                self.train_data.append({'frames': frames_path, "label": sample_label})
            else:
                sample_path = os.path.join(self.data_root, sample)
                frames_path = glob.glob('%s/*' % (sample_path))

                if len(frames_path) < 10:
                    frames_path_len_min = len(frames_path)

                frames_path.sort()
                self.test_data.append({'frames': frames_path, "label": sample_label})


        sample_list = os.listdir(self.data_root_120)
        for sample in sample_list:
            # print(sample)
            sample_name = sample.split('.')[0]
            subject_num = int(sample_name[9:12])
            sample_label = int(sample_name[17:20]) - 1

            if subject_num in self.train_subject_list:
                sample_path = os.path.join(self.data_root_120, sample)
                frames_path = glob.glob('%s/*' % (sample_path))

                if len(frames_path) < 10:
                    frames_path_len_min = len(frames_path)

                frames_path.sort()

                self.train_data.append({'frames': frames_path, "label": sample_label})
            else:
                sample_path = os.path.join(self.data_root_120, sample)
                frames_path = glob.glob('%s/*' % (sample_path))

                if len(frames_path) < 10:
                    frames_path_len_min = len(frames_path)

                frames_path.sort()
                self.test_data.append({'frames': frames_path, "label": sample_label})


        print(frames_path_len_min)
        print(len(self.train_data))
        print(len(self.test_data))

        if train:
            self.data = self.train_data
        else:
            self.data = self.test_data

        if train:
            self.clip_transform = util.clip_transform('train', clip_len)
        else:
            self.clip_transform = util.clip_transform('val', clip_len)

        self.train = train

        # print(frames_path_len_min)

    def sample(self, imgs):

        if len(imgs) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(imgs) - self.clip_len)
                imgs = imgs[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(imgs) // 2 - self.clip_len // 2
                imgs = imgs[offset:offset + self.clip_len]
                assert len(imgs) == self.clip_len, 'frame selection error!'

        imgs = [self.loader(img) for img in imgs]
        return imgs

    def sample_pth(self, pth_name):

        if self.mode == 'rgb':

            with open("%s/%s" % ("/home/data/NTUD60/rgb_pth", pth_name), 'rb') as f:
                data = pickle.load(f)
        elif self.mode == 'depth':
            with open("%s/%s" % ("/home/data/NTUD60/depth_pth", pth_name), 'rb') as f:
                data = pickle.load(f)
        else:
            data = None

        if len(data) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(data) - self.clip_len)
                data = data[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(data) // 2 - self.clip_len // 2
                data = data[offset:offset + self.clip_len]
                assert len(data) == self.clip_len, 'frame selection error!'
        return data

    def __getitem__(self, index):

        entry = self.data[index]
        # print(entry)
        a = datetime.datetime.now()
        frames = self.sample(entry['frames'])

        c = datetime.datetime.now()
        # print((c - a).total_seconds())

        frames = self.clip_transform(frames)  # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3)  # (3, T, 224, 224)
        b = datetime.datetime.now()
        # print((b-c).total_seconds())
        # print(entry['label'])
        instance = {'frames': frames, 'label': entry['label']}

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)


class NTUD120CS_MultiCrop(NTUD120_CS):

    def __init__(self, data_root, clip_len, mode, train=True, sample_interal=2):
        super(NTUD120CS_MultiCrop, self).__init__(data_root, clip_len, mode, train=train, sample_interal=sample_interal)
        self.clip_transform = util.clip_transform('3crop', self.clip_len)

    def sample(self, imgs, K=10):
        '''
        每一个动作对应的一个视频的所有帧的路径
        :param imgs:
        :param K:
        :return:
        '''

        # memoize loading images since clips overlap
        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(imgs) - self.clip_len // 2, K)]

        clips = []
        for c in centers:
            clip = imgs[c - self.clip_len // 2:c + self.clip_len // 2]
            clip = [load(img) for img in clip]
            clips.append(clip)
        return clips

    def __getitem__(self, index):

        entry = self.data[index]
        clips = self.sample(entry['frames'])

        frames = []
        for clip in clips:
            clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(3)]
            clip = torch.stack(clip, 0)  # (3, 3, 224, 224)
            frames.append(clip)
        frames = torch.stack(frames, 0)  # (10, 3, 3, 224, 224)

        instance = {'frames': frames, 'label': entry['label']}

        return instance
