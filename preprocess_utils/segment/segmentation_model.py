import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_weights_dict = dict()


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


class FCN8Model(nn.Module):

    def __init__(self, weight_file, is_full_model=False):
        super(FCN8Model, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.is_full_model = is_full_model

        self.content_vgg_conv1_1_Conv2D = self.__conv(2, name='content_vgg/conv1_1/Conv2D', in_channels=3,
                                                      out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv1_2_Conv2D = self.__conv(2, name='content_vgg/conv1_2/Conv2D', in_channels=64,
                                                      out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv2_1_Conv2D = self.__conv(2, name='content_vgg/conv2_1/Conv2D', in_channels=64,
                                                      out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv2_2_Conv2D = self.__conv(2, name='content_vgg/conv2_2/Conv2D', in_channels=128,
                                                      out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv3_1_Conv2D = self.__conv(2, name='content_vgg/conv3_1/Conv2D', in_channels=128,
                                                      out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv3_2_Conv2D = self.__conv(2, name='content_vgg/conv3_2/Conv2D', in_channels=256,
                                                      out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv3_3_Conv2D = self.__conv(2, name='content_vgg/conv3_3/Conv2D', in_channels=256,
                                                      out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv4_1_Conv2D = self.__conv(2, name='content_vgg/conv4_1/Conv2D', in_channels=256,
                                                      out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_score_pool3_Conv2D = self.__conv(2, name='content_vgg/score_pool3/Conv2D', in_channels=256,
                                                          out_channels=6, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                          bias=True)
        self.content_vgg_conv4_2_Conv2D = self.__conv(2, name='content_vgg/conv4_2/Conv2D', in_channels=512,
                                                      out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv4_3_Conv2D = self.__conv(2, name='content_vgg/conv4_3/Conv2D', in_channels=512,
                                                      out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv5_1_Conv2D = self.__conv(2, name='content_vgg/conv5_1/Conv2D', in_channels=512,
                                                      out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_score_pool4_Conv2D = self.__conv(2, name='content_vgg/score_pool4/Conv2D', in_channels=512,
                                                          out_channels=6, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                          bias=True)
        self.content_vgg_conv5_2_Conv2D = self.__conv(2, name='content_vgg/conv5_2/Conv2D', in_channels=512,
                                                      out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_conv5_3_Conv2D = self.__conv(2, name='content_vgg/conv5_3/Conv2D', in_channels=512,
                                                      out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                      bias=True)
        self.content_vgg_fc6_Conv2D = self.__conv(2, name='content_vgg/fc6/Conv2D', in_channels=512, out_channels=4096,
                                                  kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.content_vgg_fc7_Conv2D = self.__conv(2, name='content_vgg/fc7/Conv2D', in_channels=4096, out_channels=4096,
                                                  kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.content_vgg_score_fr_Conv2D = self.__conv(2, name='content_vgg/score_fr/Conv2D', in_channels=4096,
                                                       out_channels=6, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                       bias=True)

        if is_full_model:
            self.content_vgg_upscore2_conv2d_transpose = self.__transponse_conv(2,
                                                                                name='content_vgg/upscore2/conv2d_transpose',
                                                                                in_channels=6, out_channels=6,
                                                                                kernel_size=(4, 4), stride=(2, 2),
                                                                                padding=(1, 1), bias=True)
            self.content_vgg_upscore4_conv2d_transpose = self.__transponse_conv(2,
                                                                                name='content_vgg/upscore4/conv2d_transpose',
                                                                                in_channels=6, out_channels=6,
                                                                                kernel_size=(4, 4), stride=(2, 2),
                                                                                padding=(1, 1), bias=True)
            self.content_vgg_upscore32_conv2d_transpose = self.__transponse_conv(2,
                                                                                 name='content_vgg/upscore32/conv2d_transpose',
                                                                                 in_channels=6, out_channels=6,
                                                                                 kernel_size=(16, 16), stride=(8, 8),
                                                                                 padding=(4, 4), bias=True)

    def forward(self, x):

        content_vgg_conv1_1_Conv2D_pad = F.pad(x, (1, 1, 1, 1))
        content_vgg_conv1_1_Conv2D = self.content_vgg_conv1_1_Conv2D(content_vgg_conv1_1_Conv2D_pad)
        content_vgg_conv1_1_Relu = F.relu(content_vgg_conv1_1_Conv2D)

        content_vgg_conv1_2_Conv2D_pad = F.pad(content_vgg_conv1_1_Relu, (1, 1, 1, 1))
        content_vgg_conv1_2_Conv2D = self.content_vgg_conv1_2_Conv2D(content_vgg_conv1_2_Conv2D_pad)
        content_vgg_conv1_2_Relu = F.relu(content_vgg_conv1_2_Conv2D)

        content_vgg_pool1, content_vgg_pool1_idx = F.max_pool2d(content_vgg_conv1_2_Relu, kernel_size=(2, 2),
                                                                stride=(2, 2), padding=0, ceil_mode=False,
                                                                return_indices=True)

        content_vgg_conv2_1_Conv2D_pad = F.pad(content_vgg_pool1, (1, 1, 1, 1))
        content_vgg_conv2_1_Conv2D = self.content_vgg_conv2_1_Conv2D(content_vgg_conv2_1_Conv2D_pad)
        content_vgg_conv2_1_Relu = F.relu(content_vgg_conv2_1_Conv2D)

        content_vgg_conv2_2_Conv2D_pad = F.pad(content_vgg_conv2_1_Relu, (1, 1, 1, 1))
        content_vgg_conv2_2_Conv2D = self.content_vgg_conv2_2_Conv2D(content_vgg_conv2_2_Conv2D_pad)
        content_vgg_conv2_2_Relu = F.relu(content_vgg_conv2_2_Conv2D)

        content_vgg_pool2, content_vgg_pool2_idx = F.max_pool2d(content_vgg_conv2_2_Relu, kernel_size=(2, 2),
                                                                stride=(2, 2), padding=0, ceil_mode=False,
                                                                return_indices=True)

        content_vgg_conv3_1_Conv2D_pad = F.pad(content_vgg_pool2, (1, 1, 1, 1))
        content_vgg_conv3_1_Conv2D = self.content_vgg_conv3_1_Conv2D(content_vgg_conv3_1_Conv2D_pad)
        content_vgg_conv3_1_Relu = F.relu(content_vgg_conv3_1_Conv2D)

        content_vgg_conv3_2_Conv2D_pad = F.pad(content_vgg_conv3_1_Relu, (1, 1, 1, 1))
        content_vgg_conv3_2_Conv2D = self.content_vgg_conv3_2_Conv2D(content_vgg_conv3_2_Conv2D_pad)
        content_vgg_conv3_2_Relu = F.relu(content_vgg_conv3_2_Conv2D)

        content_vgg_conv3_3_Conv2D_pad = F.pad(content_vgg_conv3_2_Relu, (1, 1, 1, 1))
        content_vgg_conv3_3_Conv2D = self.content_vgg_conv3_3_Conv2D(content_vgg_conv3_3_Conv2D_pad)
        content_vgg_conv3_3_Relu = F.relu(content_vgg_conv3_3_Conv2D)

        content_vgg_pool3, content_vgg_pool3_idx = F.max_pool2d(content_vgg_conv3_3_Relu, kernel_size=(2, 2),
                                                                stride=(2, 2), padding=0, ceil_mode=False,
                                                                return_indices=True)

        content_vgg_conv4_1_Conv2D_pad = F.pad(content_vgg_pool3, (1, 1, 1, 1))
        content_vgg_conv4_1_Conv2D = self.content_vgg_conv4_1_Conv2D(content_vgg_conv4_1_Conv2D_pad)

        content_vgg_score_pool3_Conv2D = self.content_vgg_score_pool3_Conv2D(content_vgg_pool3)
        content_vgg_conv4_1_Relu = F.relu(content_vgg_conv4_1_Conv2D)

        content_vgg_conv4_2_Conv2D_pad = F.pad(content_vgg_conv4_1_Relu, (1, 1, 1, 1))
        content_vgg_conv4_2_Conv2D = self.content_vgg_conv4_2_Conv2D(content_vgg_conv4_2_Conv2D_pad)
        content_vgg_conv4_2_Relu = F.relu(content_vgg_conv4_2_Conv2D)

        content_vgg_conv4_3_Conv2D_pad = F.pad(content_vgg_conv4_2_Relu, (1, 1, 1, 1))
        content_vgg_conv4_3_Conv2D = self.content_vgg_conv4_3_Conv2D(content_vgg_conv4_3_Conv2D_pad)
        content_vgg_conv4_3_Relu = F.relu(content_vgg_conv4_3_Conv2D)

        content_vgg_pool4, content_vgg_pool4_idx = F.max_pool2d(content_vgg_conv4_3_Relu, kernel_size=(2, 2),
                                                                stride=(2, 2), padding=0, ceil_mode=False,
                                                                return_indices=True)

        content_vgg_conv5_1_Conv2D_pad = F.pad(content_vgg_pool4, (1, 1, 1, 1))
        content_vgg_conv5_1_Conv2D = self.content_vgg_conv5_1_Conv2D(content_vgg_conv5_1_Conv2D_pad)

        content_vgg_score_pool4_Conv2D = self.content_vgg_score_pool4_Conv2D(content_vgg_pool4)
        content_vgg_conv5_1_Relu = F.relu(content_vgg_conv5_1_Conv2D)

        content_vgg_conv5_2_Conv2D_pad = F.pad(content_vgg_conv5_1_Relu, (1, 1, 1, 1))
        content_vgg_conv5_2_Conv2D = self.content_vgg_conv5_2_Conv2D(content_vgg_conv5_2_Conv2D_pad)
        content_vgg_conv5_2_Relu = F.relu(content_vgg_conv5_2_Conv2D)

        content_vgg_conv5_3_Conv2D_pad = F.pad(content_vgg_conv5_2_Relu, (1, 1, 1, 1))
        content_vgg_conv5_3_Conv2D = self.content_vgg_conv5_3_Conv2D(content_vgg_conv5_3_Conv2D_pad)
        content_vgg_conv5_3_Relu = F.relu(content_vgg_conv5_3_Conv2D)

        content_vgg_pool5, content_vgg_pool5_idx = F.max_pool2d(content_vgg_conv5_3_Relu, kernel_size=(2, 2),
                                                                stride=(2, 2), padding=0, ceil_mode=False,
                                                                return_indices=True)

        content_vgg_fc6_Conv2D_pad = F.pad(content_vgg_pool5, (3, 3, 3, 3))
        content_vgg_fc6_Conv2D = self.content_vgg_fc6_Conv2D(content_vgg_fc6_Conv2D_pad)
        content_vgg_fc6_Relu = F.relu(content_vgg_fc6_Conv2D)

        content_vgg_fc7_Conv2D = self.content_vgg_fc7_Conv2D(content_vgg_fc6_Relu)
        content_vgg_fc7_Relu = F.relu(content_vgg_fc7_Conv2D)

        content_vgg_score_fr_Conv2D = self.content_vgg_score_fr_Conv2D(content_vgg_fc7_Relu)

        pred = torch.argmax(content_vgg_score_fr_Conv2D, dim=1)

        if not self.is_full_model:
            return content_vgg_score_fr_Conv2D, pred

        content_vgg_upscore2_conv2d_transpose = self.content_vgg_upscore2_conv2d_transpose(content_vgg_score_fr_Conv2D)
        content_vgg_fuse_pool_4 = content_vgg_upscore2_conv2d_transpose + content_vgg_score_pool4_Conv2D

        content_vgg_upscore4_conv2d_transpose = self.content_vgg_upscore4_conv2d_transpose(content_vgg_fuse_pool_4)
        content_vgg_fuse_pool_3 = content_vgg_upscore4_conv2d_transpose + content_vgg_score_pool3_Conv2D

        content_vgg_upscore32_conv2d_transpose = self.content_vgg_upscore32_conv2d_transpose(content_vgg_fuse_pool_3)

        pred_up = torch.argmax(content_vgg_upscore32_conv2d_transpose, dim=1)
        return pred_up

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __transponse_conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.ConvTranspose1d(**kwargs)
        elif dim == 2:
            layer = nn.ConvTranspose2d(**kwargs)
        elif dim == 3:
            layer = nn.ConvTranspose3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(np.transpose(_weights_dict[name]['weights'], (3, 2, 0, 1))))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(np.transpose(_weights_dict[name]['bias'], (3, 2, 0, 1))))
        return layer
