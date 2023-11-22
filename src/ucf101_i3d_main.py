import os
import sys

sys.path.append('..')
from configuration.config_ucf101_i3d import args
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as td
import models.resnet as resnet
import os
from lib.model_develop_utils import train_base


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    '''
    度量学习版本的retina 网络
    :param args:
    :return:
    '''

    # 初始化
    class_num = args.class_num
    args.name = args.dataset
    args.log_name = args.name + '.csv'
    args.model_name = args.name + '.pt'

    train_loader = retina_dataloader(args, train=True)
    test_loader = retina_dataloader(args, train=False)

    # define backbone and margin layer
    if args.backbone == 'i3d':
        model = resnet.i3_res50(args.class_num)
    elif args.backbone == 'i3d_nl':
        model = resnet.i3_res50_nl(args.class_num)

    else:
        print(args.backbone, ' is not available!')
        model = None

    if torch.cuda.is_available():
        model = model.cuda()

    # OPTIM
    if args.optim == "adda":

        optimizer = optim.Adam(
            {'params': filter(lambda param: param.requires_grad, model.parameters())},
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "sgd":
        optimizer = optim.SGD(
            {'params': filter(lambda param: param.requires_grad, model.parameters())},
            lr=args.lr,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    else:
        print('unrealize optim')
        optimizer = None

    criterion = nn.CrossEntropyLoss()

    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
               test_loader=test_loader,
               args=args)


if __name__ == '__main__':
    # seed_torch(2)
    main(args=args)
