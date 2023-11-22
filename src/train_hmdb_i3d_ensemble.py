'''
用于模型ensemble的测试.
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torchnet as tnt
import collections
import sys

sys.path.append('..')

from utils import util
from lib.processing_utils import save_args, makedir
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load',
                    default='')  # "/home/CVPR//shicaiwei/pytorch-resnet3d/src/cs/tmp/ntud60/rgb/16/ckpt_E_89_I_1.pth"
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_iter', default=4000000, type=int)
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--class_num', default=51)

parser.add_argument('--clip_len', type=int, default=16)
parser.add_argument('--dataset', type=str, default='hmdb51')
parser.add_argument('--freeze', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--rgb_pretrain_dir', type=str,
                    default="../results/hmdb51_0.01/0/rgb/16/ckpt_E_94_I_1.pth")
parser.add_argument('--flow_pretrain_dir', type=str,
                    default="../results/hmdb51_0.01/0/flow/16/ckpt_E_97_I_2.pth")

args = parser.parse_args()
args.mode = 'ensemble'
args.cv_dir = os.path.join("../results", args.dataset + "_" + str(args.lr), str(args.freeze), args.mode + '_fine-tune',
                           str(args.clip_len))
# if args.mode == 'flow':
#     args.lr = 1e-2
#     args.clip_len = 64
print(args)
makedir(args.cv_dir)

save_args(args, args.cv_dir, "args.txt")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


def save(epoch, iteration):
    '''
    保存模型和args
    :param epoch:
    :param iteration:
    :return:
    '''
    print('Saving state, iter:', iteration)
    state_dict = rgb_flow_net.state_dict() if not args.parallel else rgb_flow_net.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net': state_dict, 'optimizer': optim_state, 'args': args, 'iter': iteration}
    torch.save(checkpoint, '%s/ckpt_E_%d_I_%d.pth' % (args.cv_dir, epoch, iteration))


def test(ensemble_model, test_dataloader):
    ensemble_model.eval()
    testloader = test_dataloader

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        rgb_batch = {"frames": batch["frames"], 'label': batch["label"]}
        flow_batch = {"flow": batch["flow"], 'label': batch["label"]}

        if torch.cuda.is_available():
            rgb_batch = util.batch_cuda(rgb_batch)
            flow_batch = util.batch_cuda(flow_batch)

        pred, rgb_loss_dict, flow_loss_dict = ensemble_model(rgb_batch, flow_batch)

        rgb_loss_dict = {k: v.mean() for k, v in rgb_loss_dict.items() if v.numel() > 0}
        rgb_loss = sum(rgb_loss_dict.values())

        flow_loss_dict = {k: v.mean() for k, v in flow_loss_dict.items() if v.numel() > 0}
        flow_loss = sum(flow_loss_dict.values())

        for k, v in rgb_loss_dict.items():
            loss_meters[k].add(v.item())

        for k, v in flow_loss_dict.items():
            loss_meters[k].add(v.item())

        pred = pred.cpu()
        print(pred.is_cuda, batch['label'].is_cuda)

        prec_scores = util.accuracy(pred, batch['label'], topk=topk)
        for k, prec in zip(topk, prec_scores):
            loss_meters['P%s' % k].add(prec.item(), pred.shape[0])

        stats = ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
        print('%d/%d.. %s' % (idx, len(testloader), stats))

    print('(test) %s' % stats)

    for k, v in loss_meters.items():
        if k == 'P1':
            socre = v.value()[0]
    return socre


def train(iteration=0):
    rgb_flow_net.train()

    total_iters = len(trainloader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,
                                                                          600])

    average_best = 0
    average_acc = 0
    score_best = 0

    while iteration <= args.max_iter:

        with torch.no_grad():
            score = test(ensemble_model=rgb_flow_net, test_dataloader=testloader)
            print(score)
            if score > score_best:
                score_best = score
                save(epoch, 1)

        rgb_flow_net.train()

        for batch in trainloader:

            rgb_batch = {"frames": batch["frames"], 'label': batch["label"]}
            flow_batch = {"flow": batch["flow"], 'label': batch["label"]}

            if torch.cuda.is_available():
                rgb_batch = util.batch_cuda(rgb_batch)
                flow_batch = util.batch_cuda(flow_batch)

            pred, rgb_loss_dict, flow_loss_dict = rgb_flow_net(rgb_batch, flow_batch)

            rgb_loss_dict = {k: v.mean() for k, v in rgb_loss_dict.items() if v.numel() > 0}
            rgb_loss = sum(rgb_loss_dict.values())

            flow_loss_dict = {k: v.mean() for k, v in flow_loss_dict.items() if v.numel() > 0}
            flow_loss = sum(flow_loss_dict.values())

            for k, v in rgb_loss_dict.items():
                loss_meters[k].add(v.item())

            for k, v in flow_loss_dict.items():
                loss_meters[k].add(v.item())

            loss = rgb_loss + flow_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred.max(1)
            # print(pred_idx)
            batch['label'] = batch['label'].cuda()
            correct = (pred_idx == batch['label']).float().sum()
            batch_acc = correct / pred.shape[0]
            average_acc += batch_acc.cpu().numpy()
            # print(batch_acc)
            loss_meters['bAcc'].add(batch_acc.item())

            loss_meters['total_loss'].add(loss.item())

            # print(iteration)

            if iteration % args.print_every == 0:
                log_str = 'iter: %d (%d + %d/%d) | ' % (iteration, epoch, iteration % total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
                print(log_str)

            if iteration % plot_every == 0:
                for key in loss_meters:
                    writer.add_scalar('train/%s' % key, loss_meters[key].value()[0], int(100 * iteration / total_iters))

            iteration += 1

        epoch += 1
        cos_scheduler.step(epoch)
        print(epoch, optimizer.param_groups[0]['lr'], iteration, epoch)

        acc_average = average_acc / len(trainloader)
        print(acc_average)
        if acc_average > average_best and epoch > 0:
            average_best = acc_average
            save(epoch, 1)
        average_acc = 0

        if epoch % args.save_every == 0 and epoch > 0:
            save(epoch, 0)

        if epoch > 100:
            sys.exit(0)


# ----------------------------------------------------------------------------------------------------------------------------------------#
from data import hmdb51
from models import model_ensemble
import torch.optim as optim
from tensorboardX import SummaryWriter

writer = SummaryWriter('%s/tb.log' % args.cv_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
##################rgb+flow#######################

clip_len = int(args.rgb_pretrain_dir.split('/')[-2])
args.clip_len = clip_len

rgb_flow_net = model_ensemble.Ensemble_I3D(rgb_pretrain_dir=args.rgb_pretrain_dir,
                                           flow_pretain_dir=args.flow_pretrain_dir,
                                           class_num=args.class_num)

if torch.cuda.is_available():
    rgb_flow_net = rgb_flow_net.cuda()
    if args.parallel:
        rgb_flow_net = nn.DataParallel(rgb_flow_net)

nwucla_trainset = hmdb51.HMDB51(data_root="/home/ssd/video_action/hmdb51/rawframes_resize",
                                split_path="../data/split_divide/hmdb/train_list1.txt",
                                clip_len=args.clip_len,
                                mode='all',
                                sample_interal=1)

nwucla_testset = hmdb51.HMDB51(data_root="/home/ssd/video_action/hmdb51/rawframes_resize",
                               clip_len=args.clip_len,
                               split_path="../data/split_divide/hmdb/test_list1.txt",
                               mode='all',
                               sample_interal=1)

trainloader = torch.utils.data.DataLoader(nwucla_trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False)

testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers,
                                         pin_memory=False)

optim_params = list(filter(lambda p: p.requires_grad, rgb_flow_net.parameters()))
print('Optimizing %d paramters' % len(optim_params))
optimizer = optim.SGD(optim_params, lr=args.lr, weight_decay=args.weight_decay)

train()
