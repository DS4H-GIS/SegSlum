from __future__ import print_function

# import sys
# sys.path.append('utils_whitening')

import argparse
import os
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import model.backbone.utils_whitening.batch_norm as batch_norm
import model.backbone.utils_whitening.folder as folder
import model.backbone.utils_whitening.consensus_loss as consensus_loss
import model.backbone.utils_whitening.whitening as whitening


__all__ = ['ResNet', 'resnet50_white']

"""
File modified from:
    https://github.com/roysubhankar/dwt-domain-adaptation/blob/master/resnet50_dwt_mec_officehome.py
"""

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class whitening_scale_shift(nn.Module):
    def __init__(self, planes, group_size, running_mean, running_variance, track_running_stats=True, affine=True):
        super(whitening_scale_shift, self).__init__()
        self.planes = planes
        self.group_size = group_size
        self.track_running_stats = track_running_stats
        self.affine = affine
        self.running_mean = running_mean
        self.running_variance = running_variance
        
        self.wh = whitening.WTransform2d(self.planes, 
                                         self.group_size, 
                                         running_m=self.running_mean, 
                                         running_var=self.running_variance, 
                                         track_running_stats=self.track_running_stats)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.planes, 1, 1))
            self.beta = nn.Parameter(torch.zeros(self.planes, 1, 1))

    def forward(self, x):
        print("var: ", self.running_variance.shape)
        print("mean: ",self.running_mean.shape)
        out = self.wh(x)
        if self.affine:
            out = out * self.gamma + self.beta
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer, sub_layer, bn_dict, group_size=4, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = conv1x1(inplanes, planes, stride)
        if layer == 1:
            self.bns1 = whitening_scale_shift(planes=planes, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_var'],
                                              affine=False)
            self.bnt1 = whitening_scale_shift(planes=planes, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_var'],
                                              affine=False)
            self.bnt1_aug = whitening_scale_shift(planes=planes, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_var'],
                                              affine=False)
            self.gamma1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.weight'])
            self.beta1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.bias'])
        else:
            self.bns1 = batch_norm.BatchNorm2d(num_features=planes,
                                              running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_mean'],
                                              running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_var'],
                                              affine=False)
            self.bnt1 = batch_norm.BatchNorm2d(num_features=planes,
                                              running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_mean'],
                                              running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_var'],
                                              affine=False)
            self.bnt1_aug = batch_norm.BatchNorm2d(num_features=planes,
                                              running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_mean'],
                                              running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_var'],
                                              affine=False)
            self.gamma1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.weight'].view(-1, 1, 1))
            self.beta1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.bias'].view(-1, 1, 1))

        self.conv2 = conv3x3(planes, planes, stride, dilation)
        if layer == 1:
            self.bns2 = whitening_scale_shift(planes=planes, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_var'],
                                              affine=False)
            self.bnt2 = whitening_scale_shift(planes=planes, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_var'],
                                              affine=False)
            self.bnt2_aug = whitening_scale_shift(planes=planes, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_var'],
                                              affine=False)
            self.gamma2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.weight'])
            self.beta2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.bias'])
        else:
            self.bns2 = batch_norm.BatchNorm2d(num_features=planes,
                                               running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_mean'],
                                               running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_var'],
                                               affine=False)
            self.bnt2 = batch_norm.BatchNorm2d(num_features=planes,
                                               running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_mean'],
                                               running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_var'],
                                               affine=False)
            self.bnt2_aug = batch_norm.BatchNorm2d(num_features=planes,
                                               running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_mean'],
                                               running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_var'],
                                               affine=False)
            self.gamma2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.weight'].view(-1, 1, 1))
            self.beta2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.bias'].view(-1, 1, 1))

        self.conv3 = conv1x1(planes, planes * self.expansion)
        if layer == 1:
            self.bns3 = whitening_scale_shift(planes=planes * self.expansion, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_var'],
                                              affine=False)
            self.bnt3 = whitening_scale_shift(planes=planes * self.expansion, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_var'],
                                              affine=False)
            self.bnt3_aug = whitening_scale_shift(planes=planes * self.expansion, 
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_var'],
                                              affine=False)
            self.gamma3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.weight'])
            self.beta3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.bias'])
        else:
            self.bns3 = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                               running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_mean'],
                                               running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_var'],
                                               affine=False)
            self.bnt3 = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                               running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_mean'],
                                               running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_var'],
                                               affine=False)
            self.bnt3_aug = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                               running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_mean'],
                                               running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_var'],
                                               affine=False)
            self.gamma3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.weight'].view(-1, 1, 1))
            self.beta3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.bias'].view(-1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.downsample is not None:
            if layer == 1:
                self.downsample_bns = whitening_scale_shift(planes=planes * self.expansion, 
                                                            group_size=group_size,
                                                            running_mean=bn_dict['layer' + str(layer) + '.0.downsample.1.wh.running_mean'],
                                                            running_variance=bn_dict['layer' + str(layer) + '.0.downsample.1.wh.running_var'],
                                                            affine=False)
                self.downsample_bnt = whitening_scale_shift(planes=planes * self.expansion, 
                                                            group_size=group_size,
                                                            running_mean=bn_dict['layer' + str(layer) + '.0.downsample.1.wh.running_mean'],
                                                            running_variance=bn_dict['layer' + str(layer) + '.0.downsample.1.wh.running_var'],
                                                            affine=False)
                self.downsample_bnt_aug = whitening_scale_shift(planes=planes * self.expansion, 
                                                            group_size=group_size,
                                                            running_mean=bn_dict['layer' + str(layer) + '.0.downsample.1.wh.running_mean'],
                                                            running_variance=bn_dict['layer' + str(layer) + '.0.downsample.1.wh.running_var'],
                                                            affine=False)
                self.downsample_gamma = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample.1.wh.weight'])
                self.downsample_beta = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample.1.wh.bias'])
            else:
                self.downsample_bns = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                             running_m=bn_dict['layer' + str(layer) + '.0.downsample.1.running_mean'],
                                                             running_v=bn_dict['layer' + str(layer) + '.0.downsample.1.running_var'],
                                                             affine=False)
                self.downsample_bnt = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                             running_m=bn_dict['layer' + str(layer) + '.0.downsample.1.running_mean'],
                                                             running_v=bn_dict['layer' + str(layer) + '.0.downsample.1.running_var'],
                                                             affine=False)
                self.downsample_bnt_aug = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                             running_m=bn_dict['layer' + str(layer) + '.0.downsample.1.running_mean'],
                                                             running_v=bn_dict['layer' + str(layer) + '.0.downsample.1.running_var'],
                                                             affine=False)
                self.downsample_gamma = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample.1.weight'].view(-1, 1, 1))
                self.downsample_beta = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample.1.bias'].view(-1, 1, 1))

    def forward(self, x, train):
        if train:
            # to do
            identity = x
            out = self.conv1(x)
            out_s, out_t, out_t_dup = torch.split(out, split_size_or_sections=out.shape[0] // 3, dim=0)
            out = torch.cat((self.bns1(out_s), torch.cat((self.bnt1(out_t), self.bnt1_aug(out_t_dup)), dim=0) ), dim=0) * self.gamma1 + self.beta1
            out = self.relu(out)

            out = self.conv2(out)
            out_s, out_t, out_t_dup = torch.split(out, split_size_or_sections=out.shape[0] // 3, dim=0)
            out = torch.cat((self.bns2(out_s), torch.cat((self.bnt2(out_t), self.bnt2_aug(out_t_dup)), dim=0) ), dim=0) * self.gamma2 + self.beta2
            out = self.relu(out)

            out = self.conv3(out)
            out_s, out_t, out_t_dup = torch.split(out, split_size_or_sections=out.shape[0] // 3, dim=0)
            out = torch.cat((self.bns3(out_s), torch.cat((self.bnt3(out_t), self.bnt3_aug(out_t_dup)), dim=0) ), dim=0) * self.gamma3 + self.beta3

            if self.downsample is not None:
                identity = self.downsample(x)
                identity_s, identity_t, identity_t_dup = torch.split(identity, split_size_or_sections=identity.shape[0] // 3, dim=0)
                identity = torch.cat((self.downsample_bns(identity_s), 
                    torch.cat((self.downsample_bnt(identity_t), self.downsample_bnt_aug(identity_t_dup)), dim=0) ), dim=0) * self.downsample_gamma + self.downsample_beta

            out = out.clone() + identity
            out = self.relu(out)
        else:
            identity = x

            out = self.conv1(x)
            out = self.bnt1(out) * self.gamma1 + self.beta1 
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bnt2(out) * self.gamma2 + self.beta2
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bnt3(out) * self.gamma3 + self.beta3

            if self.downsample is not None:
                identity = self.downsample(x)
                identity = self.downsample_bnt(identity) * self.downsample_gamma + self.downsample_beta

            out = out.clone() + identity
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, state_dict, num_classes=2, zero_init_residual=False, group_size=4, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.bn_dict = compute_bn_stats(state_dict)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.conv2 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#         )
        self.bns1 = whitening_scale_shift(planes=64, 
                                          group_size=group_size,
                                          running_mean=self.bn_dict['bn1.wh.running_mean'],
                                          running_variance=self.bn_dict['bn1.wh.running_var'],
                                          affine=False)
        self.bnt1 = whitening_scale_shift(planes=64, 
                                          group_size=group_size,
                                          running_mean=self.bn_dict['bn1.wh.running_mean'],
                                          running_variance=self.bn_dict['bn1.wh.running_var'],
                                          affine=False)
        self.bnt1_aug = whitening_scale_shift(planes=64, 
                                          group_size=group_size,
                                          running_mean=self.bn_dict['bn1.wh.running_mean'],
                                          running_variance=self.bn_dict['bn1.wh.running_var'],
                                          affine=False)
        self.gamma1 = nn.Parameter(self.bn_dict['bn1.wh.weight'])
        self.beta1 = nn.Parameter(self.bn_dict['bn1.wh.bias'])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], self.bn_dict, layer=1)
        self.layer2 = self._make_layer(block, 128, layers[1], self.bn_dict, stride=2, layer=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], self.bn_dict, stride=2, layer=3, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], self.bn_dict, stride=2, layer=4, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, bn_dict, layer=1, group_size=4, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, layer, 0, bn_dict, group_size, stride, downsample, previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer, i, bn_dict, group_size, dilation=self.dilation))

        return nn.Sequential(*layers)

    def base_forward(self, x, train=False):
        if train:
            x = self.conv1(x)
            print(x.shape)
            x_s, x_t, x_t_dup = torch.split(x, split_size_or_sections=x.shape[0]//3)
            x = torch.cat((self.bns1(x_s), torch.cat((self.bnt1(x_t), self.bnt1_aug(x_t_dup)), dim=0) ), dim=0) * self.gamma1 + self.beta1
            
#             x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_out(x)
        else:
            x = self.conv1(x)
            x = self.bnt1(x) * self.gamma1 + self.beta1
#             x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)

            c1 = self.layer1(x, train)
            c2 = self.layer2(c1, train)
            c3 = self.layer3(c2, train)
            c4 = self.layer4(c3, train)

        return c1, c2, c3, c4


def compute_bn_stats(state_dict):
    #state_dict = state_dict = torch.load(path) #'/home/sroy/.torch/models/resnet50-19c8e357.pth'

    bn_key_names = []
    for name, param in state_dict.items():
        if name.find('bn') != -1:
            bn_key_names.append(name)
        elif name.find('downsample') != -1:
            bn_key_names.append(name)

    # keeping only the batch norm specific elements in the dictionary
    bn_dict = {k: v for k, v in state_dict.items() if k in bn_key_names}
    return bn_dict


def _resnet(arch, block, layers, pretrained, **kwargs):
    if pretrained:
#         state_dict = torch.load("pretrained/%s.pth" % arch)
        state_dict_model = torch.load("../../pretrained/resnet50.pth")
#         print(state_dict_model.keys())
        modified_state_dict = {}
        for key in state_dict_model.keys():
            if "num_batches_tracked" in key:
                modified_state_dict[key] = state_dict_model[key]
            elif key.startswith("bn1."):
                mod_key = key.split(".")[0]+".wh."+key.split(".")[1]
                mod_query = state_dict_model[key]
                print(mod_query.shape, key)
                if "running" in key:
                    modified_state_dict.update({mod_key: mod_query.repeat(1, 64, 1, 1)})
                else:
                    modified_state_dict.update({mod_key: mod_query})
            elif key.startswith("layer1.0.downsample.1"):
                mod_key = key[:22]+"wh."+key[22:]
                mod_query = state_dict_model[key]
                print(mod_query.shape, key)
                if "running" in key:
                    modified_state_dict.update({mod_key: mod_query.repeat(1, mod_query.shape[0], 1, 1)})
                else:
                    modified_state_dict.update({mod_key: mod_query})
            elif key.startswith("layer1"):
                mod_key = key[:13]+"wh."+key[13:]
                mod_query = state_dict_model[key]
                print(mod_query.shape, key)
                if "running" in key:
                    modified_state_dict.update({mod_key: mod_query.repeat(1, mod_query.shape[0], 1, 1)})
                else:
                    modified_state_dict.update({mod_key: mod_query})
            else:
                modified_state_dict[key] = state_dict_model[key]
                
        model = ResNet(block, layers, modified_state_dict, **kwargs)
        model.load_state_dict(state_dict_model, strict=False)
    return model


# def resnet18(pretrained=False):
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained)


# def resnet34(pretrained=False):
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained)


def resnet50_whitening(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


# def resnet101(pretrained=False):
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
#                    replace_stride_with_dilation=[False, True, True])


# def resnet152(pretrained=False):
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
#                    replace_stride_with_dilation=[False, True, True])
