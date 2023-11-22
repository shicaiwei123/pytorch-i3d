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
from lib.processing_utils import save_args,makedir

cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')
parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load', default='')  # "/home/CVPR/shicaiwei/pytorch-resnet3d/src/cv/tmp/ckpt_E_136_I_0.pth"
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_iter', default=400000, type=int)
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=4)

parser.add_argument('--mode', type=str, default='rgb')
parser.add_argument("--split_subject", type=int, default=9)
parser.add_argument('--cv_dir', default='cs/tmp', help='Directory for saving checkpoint models')
parser.add_argument('--clip_len', type=int, default=16)

args = parser.parse_args()
args.cv_dir = os.path.join(args.cv_dir, args.mode, str(args.split_subject), str(args.clip_len))
print(args.cv_dir)
makedir(args.cv_dir)

save_args(args, args.cv_dir, "args.txt")


def save(epoch, iteration):
    '''
    保存模型和args
    :param epoch:
    :param iteration:
    :return:
    '''
    print('Saving state, iter:', iteration)
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net': state_dict, 'optimizer': optim_state, 'args': args, 'iter': iteration}
    torch.save(checkpoint, '%s/ckpt_E_%d_I_%d.pth' % (args.cv_dir, epoch, iteration))


# milestones=[300,1000,2000] 训练1000个epoch差不多了.
# 直接训练效果就能好,还是怎样?
# 不冻结,要全部训练
# cross-view 效果也还好,问题是,emmmm,要不要弄cross-view? 还是cross-subject好了,因为我其实不需要cross-view?
# 而且其他方法也是cross-subject,那么就subject吧

def train(iteration=0):
    net.train()

    total_iters = len(trainloader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,
                                                                          1000,
                                                                          2000])

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

from data import nw_ucla
from models import resnet

nwucla_trainset = nw_ucla.NWUCLA_CS(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                    split_subject=args.split_subject,
                                    clip_len=args.clip_len,
                                    mode=args.mode,
                                    train=True,
                                    sample_interal=1)

nwucla_testset = nw_ucla.NWUCLA_CS(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                   split_subject=args.split_subject,
                                   clip_len=args.clip_len,
                                   mode=args.mode,
                                   train=False,
                                   sample_interal=1)

trainloader = torch.utils.data.DataLoader(nwucla_trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False)

pretrain_dir = "../pretrained/i3d_r50_kinetics.pth"
net = resnet.i3_res50(400, pretrain_dir=pretrain_dir)

para = net.parameters()
# for p in para:
#     p.requires_grad = False

net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

# print(net)
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
