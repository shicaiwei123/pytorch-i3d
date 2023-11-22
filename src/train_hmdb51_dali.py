import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torchnet as tnt
import collections
import sys
from nvidia.dali.plugin.pytorch import LastBatchPolicy

sys.path.append('..')

from utils import util
from lib.processing_utils import save_args, makedir
from torch.utils.data import DataLoader

cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--save_every', default=2, type=float, help='fraction of an epoch to save after')
parser.add_argument('--load',
                    default='')  # "/home/CVPR//shicaiwei/pytorch-resnet3d/src/cs/tmp/ntud60/rgb/16/ckpt_E_89_I_1.pth"
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_iter', default=4000000, type=int)
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=8)

parser.add_argument('--mode', type=str, default='rgb')
parser.add_argument('--clip_len', type=int, default=16)
parser.add_argument('--dataset', type=str, default='hmdb51')
parser.add_argument('--freeze', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
args.cv_dir = os.path.join("../results", args.dataset + "_" + str(args.lr), str(args.freeze), args.mode,
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
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net': state_dict, 'optimizer': optim_state, 'args': args, 'iter': iteration}
    torch.save(checkpoint, '%s/ckpt_E_%d_I_%d.pth' % (args.cv_dir, epoch, iteration))


def test(model, test_dataloader):
    net = model
    testloader = test_dataloader
    net.eval()

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())

    for idx, batch in enumerate(testloader):

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k: v.mean() for k, v in loss_dict.items() if v.numel() > 0}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())

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
    total_iters = train_eii.data_set_len / args.batch_size
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * total_iters)
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,
                                                                          600])
    mean = torch.tensor([106.46409, 101.46796, 93.22408])
    std = torch.tensor([68.28274, 66.40727, 67.58706])
    average_best = 0
    average_acc = 0
    score_best = 0
    iter_num = 0
    while iteration <= args.max_iter:

        net.train()
        for batch in train_dali_iter:
            iter_num += 1
            batch = batch[0]
            images = [batch[f"image_{i}"] for i in range(args.clip_len)]  # N * B * h * w*3
            images = torch.stack(images)
            images = images.permute((1, 0, 2, 3, 4))
            images = images.permute((0, 4, 1, 2, 3))
            images = images.float()
            labels = batch["label"]  # bs * 1
            labels = torch.squeeze(labels, dim=1)
            labels = labels.long().cuda()
            for i in range(images.shape[1]):
                normal = (images[:, i, :, :, :] - mean[i]) / std[i]
                images[:, i, :, :, :] = normal

            # print(images.device,labels.device)

            batch = {"frames": images, "label": labels}
            # batch = util.batch_cuda(batch)
            pred, loss_dict = net(batch)

            loss_dict = {k: v.mean() for k, v in loss_dict.items()}
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

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
                    writer.add_scalar('train/%s' % key, loss_meters[key].value()[0],
                                      int(100 * iteration / total_iters))

            iteration += 1

            if iter_num > total_iters:
                epoch += 1
                iter_num = 0
                cos_scheduler.step(epoch)
                print(epoch, optimizer.param_groups[0]['lr'], iteration, epoch)

                acc_average = average_acc
                print(acc_average)
                if acc_average > average_best and epoch > 0:
                    average_best = acc_average
                    save(epoch, 1)
                average_acc = 0

                if epoch % args.save_every == 0 and epoch > 0:
                    save(epoch, 0)

                if epoch > 250:
                    sys.exit(0)

                if epoch > 5:
                    with torch.no_grad():
                        score = test(model=net, test_dataloader=test_dali_iter)
                        print(score)
                        if score > score_best:
                            score_best = score
                            save(epoch, 2)


# ----------------------------------------------------------------------------------------------------------------------------------------#


from tensorboardX import SummaryWriter

writer = SummaryWriter('%s/tb.log' % args.cv_dir)

from data import hmdb51_dali
from models import i3dpt

start_iter = 0

train_eii = hmdb51_dali.ExternalInputIterator(data_root="/home/data/shicaiwei/video_action/rawframes_resize",
                                              split_path="../data/split_divide/hmdb/train_list1.txt", mode=args.mode,
                                              batch_size=args.batch_size, clip_len=args.clip_len, shuffled=True)
train_pipe = hmdb51_dali.ExternalSourcePipeline(external_data=train_eii, batch_size=args.batch_size,
                                                clip_len=args.clip_len, num_threads=4,
                                                device_id=0, )
train_pipe.build()

# 构建数据的含义
batch_dict = [f"image_{i}" for i in range(args.clip_len)]
batch_dict = batch_dict + ['label']
# batch_dict = ['images', 'label']

# dataloader
train_dali_iter = hmdb51_dali.DALIGenericIterator([train_pipe], batch_dict, auto_reset=True, dynamic_shape=True,
                                                  last_batch_padded=True,
                                                  last_batch_policy=LastBatchPolicy.PARTIAL)

test_eii = hmdb51_dali.ExternalInputIterator(data_root="/home/data/shicaiwei/video_action/hmdb51/rawframes_resize",
                                             split_path="../data/split_divide/hmdb/test_list1.txt", mode=args.mode,
                                             batch_size=args.batch_size, clip_len=args.clip_len, shuffled=True)
test_pipe = hmdb51_dali.ExternalSourcePipeline(external_data=test_eii, batch_size=args.batch_size,
                                               clip_len=args.clip_len, num_threads=2,
                                               device_id=0, )
test_pipe.build()

# 构建数据的含义
batch_dict = [f"image_{i}" for i in range(args.clip_len)]
batch_dict = batch_dict + ['label']
# batch_dict = ['images', 'label']

# dataloader
test_dali_iter = hmdb51_dali.DALIGenericIterator([test_pipe], batch_dict, auto_reset=True, dynamic_shape=True,
                                                 last_batch_padded=True,
                                                 last_batch_policy=LastBatchPolicy.PARTIAL)

# trainloader = DataLoaderX(trainloader)

pretrain_dir_flow = "../pretrained/model_flow.pth"
pretrain_dir_rgb = "../pretrained/model_rgb.pth"

net = i3dpt.I3D(400, modality=args.mode)
if args.mode == 'flow':
    net.load_state_dict(torch.load(pretrain_dir_flow))
elif args.mode == 'rgb':
    net.load_state_dict(torch.load(pretrain_dir_rgb))
else:
    raise RuntimeError("et.load_state_dict")

if args.freeze:
    para = net.parameters()
    for p in para:
        p.requires_grad = False

    for p in net.mixed_5b.parameters():
        p.requires_grad = True
    for p in net.mixed_5c.parameters():
        p.requires_grad = True

net.conv3d_0c_1x1 = i3dpt.Unit3Dpy(
    in_channels=1024,
    out_channels=51,
    kernel_size=(1, 1, 1),
    activation=None,
    use_bias=True,
    use_bn=False)

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

train(start_iter)
