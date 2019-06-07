"""
SST tracker net

Thanks to ssd pytorch implementation (see https://github.com/amdegroot/ssd.pytorch)
copyright: shijie Sun (shijieSun@chd.edu.cn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

config = {
    'resume': None,
    'start_iter': 55050,
    'cuda': True,
    'batch_size': 1,
    'num_workers': 16,
    'iterations': 85050,
    'learning_rate': 5e-3,
    'false_constant': 10,
    'type': 'test', # choose from ('test', 'train')
    'dataset_type': 'train', # choose from ('test', 'train')
    'detector': 'FRCNN', # choose from ('DPM', 'FRCNN', 'SDP')
    'max_object': 80,  # N
    'max_gap_frame': 40, # not the hard gap
    'min_gap_frame': 0, # not the hard gap
    'sst_dim': 900,
    'min_visibility': 0.3,
    'mean_pixel': (104, 117, 123),
    'max_expand': 1.2,
    'lower_contrast': 0.7,
    'upper_constrast': 1.5,
    'lower_saturation': 0.7,
    'upper_saturation': 1.5,
    'alpha_valid': 0.8,
    'base_net': {
        '900': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                'C', 512, 512, 512, 'M', 512, 512, 512],
        '1024': [],},
    'extra_net': {
        '900': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256,
                128, 'S', 256, 128, 256],  # new: this line
        '1024': [],
    },
    'selector_size': (255, 113, 56, 28, 14, 12, 10, 5, 3),
    'selector_channel':(60, 80, 100, 80, 60, 50, 40, 30, 20),
    'final_net' : {
        '900': [1040, 512, 256, 128, 64, 1],
        '1024': []
    },
    'vgg_source' : [15, 25, -1],
    'default_mbox': { # The default box setup
        '900': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
        '1024': [],
    }
}

# add the contraints
config['final_net']['900'][0] = np.sum(config['selector_channel'])*2


class SST(nn.Module):
    #new: combine two vgg_net
    def __init__(self, phase, base, extras, selector, final_net, use_gpu=config['cuda']):
        super(SST, self).__init__()
        self.phase = phase

        # vgg network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.selector = nn.ModuleList(selector)

        # self.vgg_next = nn.ModuleList(base_next)
        # self.extras_next = nn.ModuleList(extras_next)
        # self.selector_next = nn.ModuleList(selector_next)

        self.stacker2_bn = nn.BatchNorm2d(int(config['final_net']['900'][0]/2))
        self.final_dp = nn.Dropout(0.5)
        self.final_net = nn.ModuleList(final_net)

        self.image_size = config['sst_dim']
        self.max_object = config['max_object']
        self.selector_channel = config['selector_channel']

        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = config['false_constant']
        self.use_gpu = use_gpu

    def forward(self, x_pre, x_next, l_pre , l_next, valid_pre=None, valid_next=None):
        '''
        the sst net forward stream
        :param x_pre:  the previous image, (1, 3, 900, 900) FT
        :param x_next: the next image,  (1, 3, 900, 900) FT
        :param l_pre: the previous box center, (1, 60, 1, 1, 2) FT
        :param l_next: the next box center, (1, 60, 1, 1, 2) FT
        :param valid_pre: the previous box mask, (1, 1, 61) BT
        :param valid_next: the next box mask, (1, 1, 61) BT
        :return: the similarity matrix
        '''
        sources_pre = list()
        sources_next = list()
        x_pre = self.forward_vgg(x_pre, self.vgg, sources_pre)
        x_next = self.forward_vgg(x_next, self.vgg, sources_next)
        # x_next.register_hook(lambda grad: print('start:', grad.sum().data[0]))
        x_pre = self.forward_extras(x_pre, self.extras,
                                    sources_pre)
        x_next = self.forward_extras(x_next, self.extras,
                                     sources_next)

        x_pre = self.forward_selector_stacker1(
            sources_pre, l_pre, self.selector
        )
        x_next = self.forward_selector_stacker1(
            sources_next, l_next, self.selector
        )
        # x_pre.register_hook(lambda grad: print('selector_stacker1:', grad.sum().data[0]))
        # [B, N, N, C]
        x = self.forward_stacker2(
            x_pre, x_next
        )
        x = self.final_dp(x)
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)

        # add false unmatched row and column
        x = self.add_unmatched_dim(x)
        return x

    def forward_feature_extracter(self, x, l):
        '''
        extract features from the vgg layers and extra net
        :param x:
        :param l:
        :return: the features
        '''
        s = list()

        x = self.forward_vgg(x, self.vgg, s)
        x = self.forward_extras(x, self.extras, s)
        x = self.forward_selector_stacker1(s, l, self.selector)

        return x

    def get_similarity(self, image1, detection1, image2, detection2):
        feature1 = self.forward_feature_extracter(image1, detection1)
        feature2 = self.forward_feature_extracter(image2, detection2)
        return self.forward_stacker_features(feature1, feature2, False)


    def resize_dim(self, x, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size
        if self.use_gpu:
            new_data = Variable(torch.ones(shape)*constant).cuda()
        else:
            new_data = Variable(torch.ones(shape) * constant)
        return torch.cat([x, new_data], dim=dim)

    def forward_stacker_features_slice(self, xp, xn, fill_up_column=True, toNumpy=True):
        pre_rest_num_final = self.max_object - xp.shape[1]
        next_rest_num_final = self.max_object - xn.shape[1]
        pre_num_final = xp.shape[1]
        next_num_final = xn.shape[1]
        tmp = []
        output = []
        for i in range(0, xp.shape[1], self.max_object):
            tmp.append(xp[:, i:i+self.max_object, :])
        for xp_tmp in tmp:
            pre_rest_num = self.max_object - xp_tmp.shape[1]
            next_rest_num = self.max_object - xn.shape[1]
            pre_num = xp_tmp.shape[1]
            next_num = xn.shape[1]
            x = self.forward_stacker2(
                self.resize_dim(xp_tmp, pre_rest_num, dim=1),
                self.resize_dim(xn, next_rest_num, dim=1)
            )

            x = self.final_dp(x)
            # [B, 1, N, N]
            x = self.forward_final(x, self.final_net)
            x = x.contiguous()
            # add zero
            if next_num < self.max_object:
                x[0, 0, :, next_num:] = 0
            if pre_num < self.max_object:
                x[0, 0, pre_num:, :] = 0
            x = x[:, :, :pre_num, :next_num]
            x = x[0, 0, :]

            output.append(x)

        output = torch.cat(output, dim=0)
        # add false unmatched row and column
        output = self.resize_dim(output, 1, dim=0, constant=self.false_constant)
        output = self.resize_dim(output, 1, dim=1, constant=self.false_constant)
        x_f = F.softmax(output, dim=1)
        x_t = F.softmax(output, dim=0)
        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num_final)) + [last_row-1]
        col_slice = list(range(next_num_final)) + [last_col-1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]

        x = Variable(torch.zeros(pre_num_final, next_num_final+1).cuda())
        # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        x[0:pre_num_final, 0:next_num_final] = \
            (x_f[0:pre_num_final, 0:next_num_final] + x_t[0:pre_num_final, 0:next_num_final]) / 2.0
        x[:, next_num_final:next_num_final+1] = x_f[:pre_num_final, next_num_final:next_num_final+1]
        if fill_up_column and pre_num > 1:
            x = torch.cat([x, x[:, next_num_final:next_num_final+1].repeat(1, pre_num_final-1)], dim=1)
        if toNumpy:
            if self.use_gpu:
                y = x.data.cpu().numpy()
                # del x, x_f, x_t
                # torch.cuda.empty_cache()
            else:
                y = x.data.numpy()
            return y
        return x

    def forward_stacker_features(self, xp, xn, fill_up_column=True, toNumpy=True):

        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]
        x = self.forward_stacker2(
            self.resize_dim(xp, pre_rest_num, dim=1),
            self.resize_dim(xn, next_rest_num, dim=1)
        )

        x = self.final_dp(x)
        # [B, 1, N, N]
        x = self.forward_final(x, self.final_net)
        x = x.contiguous()
        # add zero
        if next_num < self.max_object:
            x[0, 0, :, next_num:] = 0
        if pre_num < self.max_object:
            x[0, 0, pre_num:, :] = 0
        x = x[0, 0, :]
        # add false unmatched row and column
        x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)
        x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num)) + [last_row-1]
        col_slice = list(range(next_num)) + [last_col-1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]

        x = Variable(torch.zeros(pre_num, next_num+1).cuda())
        # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        x[0:pre_num, 0:next_num] = (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        x[:, next_num:next_num+1] = x_f[:pre_num, next_num:next_num+1]
        if fill_up_column and pre_num > 1:
            x = torch.cat([x, x[:, next_num:next_num+1].repeat(1, pre_num-1)], dim=1)
        if toNumpy:
            if self.use_gpu:
                y = x.detach_().cpu().numpy()
                # del x, x_f, x_t
                # torch.cuda.empty_cache()
            else:
                y = x.detach_().numpy()
            return y
        return x

    def forward_stacker_features_differentiable(self, xp, xn, fill_up_column=True, toNumpy=True):
        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]

        x = self.forward_stacker2(
            self.resize_dim(xp, pre_rest_num, dim=1),
            self.resize_dim(xn, next_rest_num, dim=1)
        )

        x = self.final_dp(x)
        # [B, 1, N, N]
        x = self.forward_final(x, self.final_net).contiguous()
        # x = x.contiguous()
        # # add zero
        # if next_num < self.max_object:
        #     y[0, 0, :, next_num:] = x[0, 0, :, next_num:] * 0.0
        # if pre_num < self.max_object:
        #     y[0, 0, pre_num:, :] = x[0, 0, pre_num:, :] * 0.0
        # # add false unmatched row and column
        x = x.squeeze(0)
        x = x.squeeze(0)
        x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)
        x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)
        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        return x_f, x_t, \
               0.5*(x_f + x_t), pre_rest_num, next_rest_num
        #
        # # slice
        # last_row, last_col = x_f.shape
        # row_slice = list(range(pre_num)) + [last_row-1]
        # col_slice = list(range(next_num)) + [last_col-1]


        # x_f = x_f[row_slice, :]
        # x_f = x_f[:, col_slice]
        # x_t = x_t[row_slice, :]
        # x_t = x_t[:, col_slice]
        #
        #
        #
        # return (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0

        # output = Variable(torch.zeros(pre_num, next_num+1).cuda(), requires_grad=True)
        # # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        # output[0:pre_num, 0:next_num] = output[0:pre_num, 0:next_num] + (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        # output[:, next_num:next_num+1] = output[:, next_num:next_num+1] + x_f[:pre_num, next_num:next_num+1]
        # if fill_up_column and pre_num > 1:
        #     output = torch.cat([output, output[:, next_num:next_num+1].repeat(1, pre_num-1)], dim=1)
        # if toNumpy:
        #     if self.use_gpu:
        #         y = output.data.cpu().numpy()
        #         # del x, x_f, x_t
        #         # torch.cuda.empty_cache()
        #     else:
        #         y = output.data.numpy()
        #     return y
        # return output

    def forward_vgg(self, x, vgg, sources):
        for k in range(16):
            x = vgg[k](x)
        sources.append(x)

        for k in range(16, 23):
            x = vgg[k](x)
        sources.append(x)

        for k in range(23, 35):
            x = vgg[k](x)
        sources.append(x)
        return x

    def forward_extras(self, x, extras, sources):
        for k, v in enumerate(extras):
            x = v(x) #x = F.relu(v(x), inplace=True)        #done: relu is unnecessary.
            if k % 6 == 3:                  #done: should select the output of BatchNormalization (-> k%6==2)
                sources.append(x)
        return x

    def forward_selector_stacker1(self, sources, labels, selector):
        '''
        :param sources: [B, C, H, W]
        :param labels: [B, N, 1, 1, 2]
        :return: the connected feature
        '''
        sources = [
            F.relu(net(x), inplace=True) for net, x in zip(selector, sources)
        ]

        res = list()
        for label_index in range(labels.size(1)):
            label_res = list()
            for source_index in range(len(sources)):
                # [N, B, C, 1, 1]
                label_res.append(
                    # [B, C, 1, 1]
                    F.grid_sample(sources[source_index],  # [B, C, H, W]
                                  labels[:, label_index, :]  # [B, 1, 1, 2
                                  ).squeeze(2).squeeze(2)
                )
            res.append(torch.cat(label_res, 1))

        return torch.stack(res, 1)

    def forward_stacker2(self, stacker1_pre_output, stacker1_next_output):
        stacker1_pre_output = stacker1_pre_output.unsqueeze(2).repeat(1, 1, self.max_object, 1).permute(0, 3, 1, 2)
        stacker1_next_output = stacker1_next_output.unsqueeze(1).repeat(1, self.max_object, 1, 1).permute(0, 3, 1, 2)

        stacker1_pre_output = self.stacker2_bn(stacker1_pre_output.contiguous())
        stacker1_next_output = self.stacker2_bn(stacker1_next_output.contiguous())

        output = torch.cat(
            [stacker1_pre_output, stacker1_next_output],
            1
        )

        return output

    def forward_final(self, x, final_net):
        x = x.contiguous()
        for f in final_net:
            x = f(x)
        return x

    def add_unmatched_dim(self, x):
        if self.false_objects_column is None:
            self.false_objects_column = Variable(torch.ones(x.shape[0], x.shape[1], x.shape[2], 1)) * self.false_constant
            if self.use_gpu:
                self.false_objects_column = self.false_objects_column.cuda()
        x = torch.cat([x, self.false_objects_column], 3)

        if self.false_objects_row is None:
            self.false_objects_row = Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3])) * self.false_constant
            if self.use_gpu:
                self.false_objects_row = self.false_objects_row.cuda()
        x = torch.cat([x, self.false_objects_row], 2)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage)
            )
            print('Finished')
        else:
            print('Sorry only .pth and .pkl files supported.')

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=True):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                conv2d = nn.Conv2d(in_channels, cfg[k+1],
                                     kernel_size=(1, 3)[flag],
                                     stride=2,
                                     padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[k+1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v,
                                     kernel_size=(1, 3)[flag])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers

def add_final(cfg, batch_normal=True):
    layers = []
    in_channels = int(cfg[0])
    layers += []
    # 1. add the 1:-2 layer with BatchNorm
    for v in cfg[1:-2]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        if batch_normal:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    # 2. add the -2: layer without BatchNorm for BatchNorm would make the output value normal distribution.
    for v in cfg[-2:]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return layers

def selector(vgg, extra_layers, batch_normal=True):
    '''
    batch_normal must be same to add_extras batch_normal
    '''
    selector_layers = []
    vgg_source = config['vgg_source']

    for k, v in enumerate(vgg_source):
         selector_layers += [nn.Conv2d(vgg[v-1].out_channels,
                              config['selector_channel'][k],
                              kernel_size=3,
                              padding=1)]
    if batch_normal:
        for k, v in enumerate(extra_layers[3::6], 3):
            selector_layers += [nn.Conv2d(v.out_channels,
                                 config['selector_channel'][k],
                                 kernel_size=3,
                                 padding=1)]
    else:
        for k, v in enumerate(extra_layers[3::4], 3):
            selector_layers += [nn.Conv2d(v.out_channels,
                                 config['selector_channel'][k],
                                 kernel_size=3,
                                 padding=1)]

    return vgg, extra_layers, selector_layers


def build_sst(phase, size=900, use_gpu=config['cuda']):
    '''
    create the SSJ Tracker Object
    :return: ssj tracker object
    '''
    if phase != 'test' and phase != 'train':
        print('Error: Phase not recognized')
        return

    if size != 900:
        print('Error: Sorry only SST{} is supported currently!'.format(size))
        return

    base = config['base_net']
    extras = config['extra_net']
    final = config['final_net']

    return SST(phase,
               *selector(
                   vgg(base[str(size)], 3),
                   add_extras(extras[str(size)], 1024)
               ),
               add_final(final[str(size)]),
               use_gpu
               )
