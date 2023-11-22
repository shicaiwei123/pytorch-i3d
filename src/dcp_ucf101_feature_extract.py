import torch
import torch.nn as nn
import numpy as np
import argparse
import collections
import torchnet as tnt
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
sys.path.append('..')
from utils import util
from lib.processing_utils import save_csv

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--test_mode', default='clip', help='video|clip')
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
args = parser.parse_args()


def test(rgb_net, flow_net, test_dataloader):
    rgb_net.eval()
    flow_net.eval()
    testloader = test_dataloader

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        rgb_batch = {"frames": batch["frames_rgb"], 'label': batch["label"]}
        flow_batch = {"flow": batch["frames_depth"], 'label': batch["label"]}
        label_batch = batch['label']

        if torch.cuda.is_available():
            rgb_batch = util.batch_cuda(rgb_batch)
            flow_batch = util.batch_cuda(flow_batch)

        pred, loss_dict, feat_list_rgb = rgb_net(rgb_batch, True)
        pred, loss_dict, feat_list_flow = rgb_net(flow_batch, True)
        label_batch = torch.unsqueeze(label_batch, 0)
        print(len(feat_list_flow))


        # save_csv("nwucla_feat_all.csv",)


# ----------------------------------------------------------------------------------------------------------------------------------------#
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
from data import nw_ucla
from models import i3dpt

args.split_num = 3
args.clip_len = 8
args.mode = 'all'

if args.test_mode == 'video':
    nwucla_testset = nw_ucla.NWUCLACV_MultiCrop(data_root="/home/data/shicaiwei/video_action/N-UCLA/multiview_action",
                                                split_num=args.split_num,
                                                clip_len=args.clip_len,
                                                mode=args.mode,

                                                train=False,
                                                sample_interal=1)
elif args.test_mode == 'clip':

    nwucla_testset = nw_ucla.NWUCLA_CV(data_root="/home/data/shicaiwei/video_action/N-UCLA/multiview_action",
                                       split_num=args.split_num,
                                       clip_len=args.clip_len,
                                       mode=args.mode,
                                       discrete=True,
                                       train=False,
                                       sample_interal=1)

testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers)

pretrain_dir_flow = "../pretrained/model_flow.pth"
pretrain_dir_rgb = "../pretrained/model_rgb.pth"

rgb_net = i3dpt.I3D(400, modality='rgb')
flow_net = i3dpt.I3D(400, modality='flow')
rgb_net.load_state_dict(torch.load(pretrain_dir_rgb))
flow_net.load_state_dict(torch.load(pretrain_dir_flow))

rgb_net.cuda()
flow_net.cuda()

with torch.no_grad():
    test(rgb_net, flow_net, test_dataloader=testloader)










