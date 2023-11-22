import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torchnet as tnt
import collections
import sys
import torch.nn.functional as F
import numpy as np
import classify

sys.path.append('..')

from utils import util
from lib.processing_utils import save_args
from dcp_loss import instance_contrastive_Loss, category_contrastive_loss, Prediction

cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')
parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load',
                    default='')  # '/home/icml/shicaiwei/pytorch-resnet3d/src/cv/tmp/rgb/2/16/ckpt_clip2_E_53_I_1.pth
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_iter', default=400000, type=int)
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=4)

parser.add_argument('--mode', type=str, default='rgb')
parser.add_argument("--split_num", type=int, default=3)
parser.add_argument('--cv_dir', default='cv/tmp2', help='Directory for saving checkpoint models')
parser.add_argument('--clip_len', type=int, default=8)
parser.add_argument('--class_num', default=10)

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
args = parser.parse_args()
args.cv_dir = os.path.join(args.cv_dir, args.mode, str(args.split_num), str(args.clip_len))
os.makedirs(args.cv_dir, exist_ok=True)

print(args)
save_args(args, args.cv_dir, "args.txt")


# rgb origin rgb
# depth vis depth with theirself processing
# origin depth  origin depth
# same ???

# split=8 都是和原始的论文一样.为了方便对比.顺便也看看他们用的到底是不是只是camera3?


def save(epoch, iteration):
    '''
    保存模型和args
    :param epoch:
    :param iteration:
    :return:
    '''
    print('Saving state, iter:', iteration)
    state_dict = rgb_net.state_dict() if not args.parallel else rgb_net.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net': state_dict, 'optimizer': optim_state, 'args': args, 'iter': iteration}
    torch.save(checkpoint, '%s/ckpt_clip%s_E_%d_I_%d.pth' % (args.cv_dir, str(args.split_num), epoch, iteration))


# milestones=[300,1000,2000] 训练1000个epoch差不多了.
# 直接训练效果就能好,还是怎样?
# 不冻结,要全部训练
# cross-view 效果也还好,问题是,emmmm,要不要弄cross-view? 还是cross-subject好了,因为我其实不需要cross-view?
# 而且其他方法也是cross-subject,那么就subject吧


def train(iteration=0):
    rgb_net.train()
    flow_net.cuda()

    total_iters = len(trainloader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    # cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,
    #                                                                       300,
    #                                                                       500])

    average_best = 0
    average_acc = 0
    score_best = 0
    epoch = 0
    while iteration <= args.max_iter:
        print(epoch)
        for idx, batch in enumerate(trainloader):
            rgb_net.train(), flow_net.train()
            rgb2flow_transform.train(), flow2rgb_transform.train()

            rgb_batch = {"frames": batch["frames_rgb"], 'label': batch["label"]}
            flow_batch = {"flow": batch["frames_depth"], 'label': batch["label"]}

            if torch.cuda.is_available():
                rgb_batch = util.batch_cuda(rgb_batch)
                flow_batch = util.batch_cuda(flow_batch)

            pred, loss_dict_rgb, feat_list_rgb = rgb_net(rgb_batch, True)
            pred, loss_dict_flow, feat_list_flow = flow_net(flow_batch, True)

            feat_list_rgb[0] = torch.from_numpy(feat_list_rgb[0])
            feat_list_flow[0] = torch.from_numpy(feat_list_flow[0])

            if feat_list_flow[0].shape[0] <= 1:
                break

            loss_icl = instance_contrastive_Loss(feat_list_rgb[0], feat_list_flow[0], 10)

            loss_ccl = category_contrastive_loss(torch.cat([feat_list_rgb[0], feat_list_flow[0]], dim=1),
                                                 batch['label'],
                                                 args.class_num, flag_gt=False)

            rgb2flow, _ = rgb2flow_transform(feat_list_rgb[0])
            flow2rgb, _ = flow2rgb_transform(feat_list_flow[0])
            recon3 = F.mse_loss(rgb2flow, feat_list_flow[0])
            recon4 = F.mse_loss(flow2rgb, feat_list_rgb[0])
            dual_loss = (recon3 + recon4)

            loss = loss_icl + loss_ccl + dual_loss * 0.1

            print("icl:", loss_icl.cpu().detach().numpy(),"ccl:",  loss_ccl.cpu().detach().numpy(),"dual:",
                  dual_loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch += 1
        if epoch > 20:

            # Evalution
            with torch.no_grad():
                rgb_net.eval(), flow_net.eval()
                rgb2flow_transform.eval(), flow2rgb_transform.eval()
                train_list_rgb = []
                train_list_flow = []
                train_label = []
                for idx, batch in enumerate(trainloader):

                    rgb_batch = {"frames": batch["frames_rgb"], 'label': batch["label"]}
                    flow_batch = {"flow": batch["frames_depth"], 'label': batch["label"]}

                    if torch.cuda.is_available():
                        rgb_batch = util.batch_cuda(rgb_batch)
                        flow_batch = util.batch_cuda(flow_batch)

                    pred, loss_dict_rgb, feat_list_rgb = rgb_net(rgb_batch, True)
                    pred, loss_dict_flow, feat_list_flow = flow_net(flow_batch, True)
                    if feat_list_flow[0].shape[0] <= 1:
                        break

                    feat_list_rgb[0] = torch.from_numpy(feat_list_rgb[0])
                    feat_list_flow[0] = torch.from_numpy(feat_list_flow[0])
                    train_list_rgb.append(feat_list_rgb[0])
                    train_list_flow.append(feat_list_flow[0])
                    train_label.append(batch['label'])
                # Training data
                imgs_latent_eval = torch.cat(train_list_rgb, dim=0)
                txts_latent_eval = torch.cat(train_list_flow, dim=0)
                train_label = torch.cat(train_label, dim=0)
                labels_train = np.array(train_label)

                latent_code_img_eval = imgs_latent_eval
                latent_code_txt_eval = txts_latent_eval

                latent_img_train = latent_code_img_eval.cpu().numpy()
                latent_txt_train = latent_code_txt_eval.cpu().numpy()
                latent_fusion_train = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

                # Test data
                test_list_rgb = []
                test_list_flow = []
                test_label = []
                for idx, batch in enumerate(testloader):

                    rgb_batch = {"frames": batch["frames_rgb"], 'label': batch["label"]}
                    flow_batch = {"flow": batch["frames_depth"], 'label': batch["label"]}

                    if torch.cuda.is_available():
                        rgb_batch = util.batch_cuda(rgb_batch)
                        flow_batch = util.batch_cuda(flow_batch)

                    pred, loss_dict_rgb, feat_list_rgb = rgb_net(rgb_batch, True)
                    pred, loss_dict_flow, feat_list_flow = flow_net(flow_batch, True)
                    if feat_list_flow[0].shape[0] <= 1:
                        break
                    feat_list_rgb[0] = torch.from_numpy(feat_list_rgb[0])
                    feat_list_flow[0] = torch.from_numpy(feat_list_flow[0])

                    test_list_rgb.append(feat_list_rgb[0])
                    test_list_flow.append(feat_list_flow[0])
                    test_label.append(batch['label'])

                imgs_latent_eval = torch.cat(test_list_rgb, dim=0)
                txts_latent_eval = torch.cat(test_list_flow, dim=0)
                test_label = torch.cat(test_label, dim=0)
                gt_batch = np.array(test_label)

                # R->D
                latent_code_img_eval_RD = imgs_latent_eval
                txt_recover_latent_eval_RD, _ = rgb2flow_transform(imgs_latent_eval)
                latent_code_txt_eval_RD = txt_recover_latent_eval_RD
                latent_fusion_test_RD = torch.cat([latent_code_img_eval_RD, latent_code_txt_eval_RD],
                                                  dim=1).cpu().numpy()

                # D->R
                latent_code_txt_eval_DR = txts_latent_eval
                img_recover_latent_eval_DR, _ = flow2rgb_transform(txts_latent_eval)
                latent_code_img_eval_DR = img_recover_latent_eval_DR

                latent_fusion_test_DR = torch.cat([latent_code_img_eval_DR, latent_code_txt_eval_DR],
                                                  dim=1).cpu().numpy()
                # R+D
                latent_fusion_test = torch.cat([imgs_latent_eval, txts_latent_eval],
                                               dim=1).cpu().numpy()

                from sklearn.metrics import accuracy_score

                label_pre = classify.vote(latent_fusion_train, latent_fusion_test_RD, labels_train)
                scores_RD = accuracy_score(gt_batch, label_pre)

                label_pre = classify.vote(latent_fusion_train, latent_fusion_test_DR, labels_train)
                scores_DR = accuracy_score(gt_batch, label_pre)

                label_pre = classify.vote(latent_fusion_train, latent_fusion_test, labels_train)
                scores = accuracy_score(gt_batch, label_pre)

                label_pre = classify.vote(latent_img_train, imgs_latent_eval.cpu().numpy(), labels_train)
                scores_onlyrgb = accuracy_score(gt_batch, label_pre)

                label_pre = classify.vote(latent_txt_train, txts_latent_eval.cpu().numpy(), labels_train)
                scores_onlydepth = accuracy_score(gt_batch, label_pre)

                print(scores_RD, scores_DR, scores, scores_onlyrgb, scores_onlydepth)


# ----------------------------------------------------------------------------------------------------------------------------------------#

from tensorboardX import SummaryWriter

writer = SummaryWriter('%s/tb.log' % args.cv_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
from data import nw_ucla
from models import resnet

args.split_num = 3
args.clip_len = 8
args.mode = 'all'
args.train = True

nwucla_trainset = nw_ucla.NWUCLA_CV(data_root="/home/data/shicaiwei/video_action/N-UCLA/multiview_action",
                                    split_num=args.split_num,
                                    clip_len=args.clip_len,
                                    mode=args.mode,
                                    train=True,
                                    sample_interal=1)

nwucla_testset = nw_ucla.NWUCLA_CV(data_root="/home/data/shicaiwei/video_action/N-UCLA/multiview_action",
                                   split_num=args.split_num,
                                   clip_len=args.clip_len,
                                   mode=args.mode,
                                   train=False,
                                   sample_interal=1)

trainloader = torch.utils.data.DataLoader(nwucla_trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False, drop_last=True)

testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=False, drop_last=True)

pretrain_dir = "../pretrained/i3d_r50_kinetics.pth"

args.mode = 'rgb'
rgb_net = resnet.i3_res50(400, pretrain_dir, args)
args.mode = 'flow'
flow_net = resnet.i3_res50(400, pretrain_dir, args)

rgb_net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
flow_net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

rgb2flow_transform = Prediction([2048, 128, 256, 128])
flow2rgb_transform = Prediction([2048, 128, 256, 128])

# print(net)
rgb_net.cuda()
flow_net.cuda()

optim_params = list(filter(lambda p: p.requires_grad, rgb_net.parameters())) + list(
    filter(lambda p: p.requires_grad, flow_net.parameters())) + list(
    filter(lambda p: p.requires_grad, rgb2flow_transform.parameters())) + list(
    filter(lambda p: p.requires_grad, flow2rgb_transform.parameters()))
print('Optimizing %d paramters' % len(optim_params))
optimizer = optim.SGD(optim_params, lr=args.lr, weight_decay=args.weight_decay)

if args.load:
    checkpoint = torch.load(args.load, map_location='cpu')
    start_iter = checkpoint['iter']
    rgb_net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Loaded checkpoint from %s' % os.path.basename(args.load))

start_iter = 0

train(start_iter)
