import torch.nn as nn
import torch
from models.resnet import i3_res50
from models.i3dpt import I3D
from models import i3dpt


class Ensemble(nn.Module):
    '''
    depth i3d 和rgb i3d 的集成
    不局限数据集和测试协议,输入不同的配对的模型就可以.
    '''

    def __init__(self, rgb_pretrain_dir, depth_pretain_dir, class_num):
        super(Ensemble, self).__init__()
        self.rgb_model = i3_res50(num_classes=class_num, pretrain_dir=rgb_pretrain_dir)
        self.depth_model = i3_res50(num_classes=class_num, pretrain_dir=depth_pretain_dir)

    def forward(self, rgb_batch, depth_batch):
        rgb_pred, rgb_loss_dict = self.rgb_model(rgb_batch)
        depth_pred, depth_loss_dict = self.depth_model(depth_batch)

        pred = (rgb_pred + depth_pred) / 2.0
        return pred, rgb_loss_dict, depth_loss_dict


class Ensemble_I3D(nn.Module):
    '''
    flow i3d 和rgb i3d 的集成
    不局限数据集和测试协议,输入不同的配对的模型就可以.
    '''

    def __init__(self, rgb_pretrain_dir, flow_pretain_dir, class_num, freeze=True):
        super(Ensemble_I3D, self).__init__()
        self.rgb_model = I3D(num_classes=class_num, modality='rgb')
        self.rgb_model.load_state_dict(torch.load(rgb_pretrain_dir)['net'])

        self.flow_model = I3D(num_classes=class_num, modality='flow')
        self.flow_model.load_state_dict(torch.load(flow_pretain_dir)['net'])

        if freeze:
            for p in self.rgb_model.parameters():
                p.requires_grad = False
            for p in self.flow_model.parameters():
                p.requires_grad = False

        self.rgb_model.conv3d_0c_1x1 = i3dpt.Unit3Dpy(
            in_channels=1024,
            out_channels=class_num,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

        self.flow_model.conv3d_0c_1x1 = i3dpt.Unit3Dpy(
            in_channels=1024,
            out_channels=class_num,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    def forward(self, rgb_batch, flow_batch):
        rgb_pred, rgb_loss_dict = self.rgb_model(rgb_batch)
        flow_pred, flow_loss_dict = self.flow_model(flow_batch)

        pred = rgb_pred * 0.5 + flow_pred * 0.5
        return pred, rgb_loss_dict, flow_loss_dict
