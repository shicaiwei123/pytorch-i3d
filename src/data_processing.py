from __future__ import print_function

import os
import sys
import glob
import argparse
from pipes import quote
from multiprocessing import Pool, current_process

import cv2
import numpy as np
import sys

sys.path.append('..')
from lib.processing_utils import get_file_list

import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image


def run_optical_flow(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(
        df_path + 'build/extract_gpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0],
        new_size[1])

    os.system(cmd)
    print('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


def get_optical_flow():
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--src_dir", type=str, default='./UCF-101',
                        help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='./ucf101_frames',
                        help='path to store frames and optical flow')
    parser.add_argument("--df_path", type=str, default='./dense_flow/',
                        help='path to the dense_flow toolbox')

    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')

    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--num_gpu", type=int, default=2, help='number of GPU')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir', 'zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi', 'mp4'],
                        help='video file extensions')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print("creating folder: " + out_path)
        os.makedirs(out_path)

    vid_list = glob.glob(src_path + '/*/*.' + ext)
    print(len(vid_list))
    pool = Pool(num_worker)
    pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))


def process_ntud_depth():
    ntud_depth_dir = "/home/data/NTUD60/Depth"
    depth_img_path_list = get_file_list(ntud_depth_dir)
    count_num = 0
    for path in depth_img_path_list:
        print(count_num)
        count_num += 1
        img = cv2.imread(path)
        if img is None:
            print(path)
            continue
        img_processed = np.uint8(255 * (img / np.max(img)))
        cv2.imwrite(path, img_processed)

    # ntud_rgb_dir = "/home/data/shicaiwei/NTUD60/nturgb+d_depth_masked"
    # ntud_rgb_path_list = get_file_list(ntud_rgb_dir)
    # ntud_rgb_path_list.sort()
    # print(len(ntud_rgb_path_list))
    # ntud_rgb_path_list = ntud_rgb_path_list[1499999:2000000]


def img_resize():
    # ntud_rgb_dir = "/home/data/NTUD60/rawframes"
    # ntud_rgb_path_list = get_file_list(ntud_rgb_dir)
    # ntud_rgb_path_list.sort()
    # ntud_rgb_path_list = ntud_rgb_path_list[3550000:4000000]

    ntud_rgb_dir = "/home/data/shicaiwei/NTUD60/ntud_depth_120/nturgb+d_depth_masked_s026"
    ntud_rgb_path_list = get_file_list(ntud_rgb_dir)

    ntud_rgb_path_list.sort(reverse=True)
    print(len(ntud_rgb_path_list))
    ntud_rgb_path_list = ntud_rgb_path_list

    # ntud_rgb_path_list.sort(reverse=True)
    # ntud_rgb_path_list = ntud_rgb_path_list[len(ntud_rgb_path_list)-1000000:len(ntud_rgb_path_list)-750000]

    count_num = 0
    import datetime
    e = datetime.datetime.now()
    for path in ntud_rgb_path_list:
        count_num += 1

        a = datetime.datetime.now()
        print(path)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        img = np.uint8(255 * (img / np.max(img)))
        print(count_num)
        cv2.imwrite(path, img)

        b = datetime.datetime.now()
        print((b - a).total_seconds())

    c = datetime.datetime.now()
    print((c - e).total_seconds())


if __name__ == '__main__':
    img_resize()
