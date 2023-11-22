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

cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')
parser.add_argument('--cv_dir', default='cv/tmp', help='Directory for saving checkpoint models')
parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load', default='')#'/home/CVPR//shicaiwei/pytorch-resnet3d/src/cv/tmp/ckpt_E_70_I_1.pth'
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_iter', default=400000, type=int)
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=4)

parser.add_argument('--clip_len', type=int, default=32)

args = parser.parse_args()
args.cv_dir = os.path.join(args.cv_dir, str(args.clip_len))

os.makedirs(args.cv_dir, exist_ok=True)


def save(epoch, iteration):
    print('Saving state, iter:', iteration)
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net': state_dict, 'optimizer': optim_state, 'args': args, 'iter': iteration}
    torch.save(checkpoint, '%s/ckpt_E_%d_I_%d.pth' % (args.cv_dir, epoch, iteration))


def train(iteration=0):
    net.train()

    total_iters = len(trainloader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,
                                                                          300,
                                                                          400])

    average_best = 0
    average_acc = 0
    while iteration <= args.max_iter:

        for batch in trainloader:

            batch = util.batch_cuda(batch)
            pred, loss_dict = net(batch)

            loss_dict = {k: v.mean() for k, v in loss_dict.items()}
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred.max(1)
            # print(pred_idx)
            correct = (pred_idx == batch['label']).float().sum()
            batch_acc = correct / pred.shape[0]
            average_acc += batch_acc.cpu().numpy()
            # print(batch_acc)
            loss_meters['bAcc'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
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
        if acc_average > average_best:
            average_best = acc_average
            save(epoch, 1)
        average_acc = 0

        if epoch % args.save_every == 0:
            save(epoch, 0)

        # ----------------------------------------------------------------------------------------------------------------------------------------#


from tensorboardX import SummaryWriter

writer = SummaryWriter('%s/tb.log' % args.cv_dir)

from data import kinetics, ucf101
from models import resnet

# trainset = kinetics.Kinetics(root='data/kinetics/', split='train', clip_len=32)

UCF101_trainset = ucf101.UCF101(data_root="/home/data/ucf101/rawframes",
                                train_split_path="/home/data/ucf101/ucfTrainTestlist/trainlist01.txt",
                                test_split_path="/home/data/ucf101/ucfTrainTestlist/testlist01.txt",
                                clip_len=args.clip_len,
                                train=True,
                                sample_interal=1)

UCF101_testset = ucf101.UCF101(data_root="/home/data/ucf101/rawframes",
                               train_split_path="/home/data/ucf101/ucfTrainTestlist/trainlist01.txt",
                               test_split_path="/home/data/ucf101/ucfTrainTestlist/testlist01.txt",
                               clip_len=args.clip_len,
                               train=False,
                               sample_interal=1)

trainloader = torch.utils.data.DataLoader(UCF101_trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False)

pretrain_dir = "../pretrained/i3d_r50_kinetics.pth"
net = resnet.i3_res50(400, pretrain_dir=pretrain_dir)

# para = net.parameters()
# for p in para:
#     p.requires_grad = False

net.fc = nn.Linear(in_features=2048, out_features=101, bias=True)

print(net)
net.cuda()

optim_params = list(filter(lambda p: p.requires_grad, net.parameters()))
print('Optimizing %d paramters' % len(optim_params))
optimizer = optim.SGD(optim_params, lr=args.lr, weight_decay=args.weight_decay)

if args.load:
    checkpoint = torch.load(args.load, map_location='cpu')
    start_iter = checkpoint['iter']
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Loaded checkpoint from %s' % os.path.basename(args.load))

start_iter = 0

train(start_iter)
