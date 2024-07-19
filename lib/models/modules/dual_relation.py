from cmath import sqrt
from turtle import forward
from .se_block import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


BN_MOMENTUM = 0.1
class InstanceRelationModule(nn.Module):
    def __init__(self, cfg, in_plane, out_plane=32):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self.inst_input_channels = in_plane

        self.inst_feat_dim = cfg.DATASET.OUTPUT_SIZE
        self.kpts_num = cfg.DATASET.NUM_KEYPOINTS
        self.out_plane = out_plane
                        
        self.norm_fact = self.inst_feat_dim
    
    def forward(self, instance_feats, imgid, instance_param, j, type=None):
        # instance_feats: [p, 32, 128, 128], kpt_feats: [p, kpts, 128, 128]
        p, c, h, w = instance_feats.size()
        residual_feats = instance_feats
        imgid = imgid.cpu().numpy().tolist()
        dic, inst_feats = {}, None

        x = instance_feats.permute(1, 0, 2, 3).contiguous()
        x = x.view(c, p, -1).contiguous()

        for key in imgid:
            dic[key] = dic.get(key, 0) + 1
        start = 0
        dic_idx = list(dic.keys())
        for i in dic.keys():
            if start+dic[i] > len(imgid):
                break
            if dic_idx.index(i) == 0:
                features = x[:, :dic[i], :]
                inst_param = instance_param[:dic[i], :]
                start = dic[i]
            else:
                features = x[:, start:start+dic[i], :]
                inst_param = instance_param[start:start+dic[i], :]  
                start = start + dic[i]
            # print(inst_feats.shape, features.shape)
            instance_relation = torch.matmul(features, features.permute(0, 2, 1).contiguous()) / self.norm_fact  # [p, p]
            inst_param_relation = torch.matmul(inst_param, inst_param.permute(1, 0).contiguous()) / 480.0
            instance_relation += inst_param_relation
            if type == 'ij':
                np.save('/data2_12t/user/dyh/results/DHRNet/inst_relation/ij_inst_{}_{}.npy'.format(str(j), i), instance_relation.cpu().detach().numpy())
                np.save('/data2_12t/user/dyh/results/DHRNet/inst_relation/ij_inst_pos_{}_{}.npy'.format(str(j), i), inst_param_relation.cpu().detach().numpy())
                np.save('/data2_12t/user/dyh/results/DHRNet/inst_relation/ij_inst_att_{}_{}.npy'.format(str(j), i), instance_relation.cpu().detach().numpy())
            else:
                np.save('/data2_12t/user/dyh/results/DHRNet/inst_relation/ji_inst_{}_{}.npy'.format(str(j), i), instance_relation.cpu().detach().numpy())
                np.save('/data2_12t/user/dyh/results/DHRNet/inst_relation/ji_inst_pos_{}_{}.npy'.format(str(j), i), inst_param_relation.cpu().detach().numpy())
                np.save('/data2_12t/user/dyh/results/DHRNet/inst_relation/ji_inst_att_{}_{}.npy'.format(str(j), i), instance_relation.cpu().detach().numpy())
            
            instance_relation = F.softmax(instance_relation, dim=0)
            instance_relation_feat = torch.matmul(instance_relation, features)
            if dic_idx.index(i) == 0:
                inst_feats = instance_relation_feat
            if dic_idx.index(i) != 0:
                inst_feats = torch.cat([inst_feats, instance_relation_feat], dim=1)
        # print(inst_feats.shape)
        instance_relation = inst_feats.view(c, p, h, w).permute(1, 0, 2, 3).contiguous()
        instance_relation += residual_feats
        return  F.relu(instance_relation) # [p, 32, 128, 128]


class JointRelationModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.inplanes = cfg.MODEL.GFD.CHANNELS
        self.kpts_num = cfg.DATASET.NUM_KEYPOINTS
        self.fea_dim = cfg.DATASET.OUTPUT_SIZE

        # self.joint_conv = nn.Conv2d(self.inplanes, self.kpts_num, 3, 1, 1)

        self.kpts_conv_k = nn.Conv2d(self.kpts_num, self.kpts_num, kernel_size=(1, 1), stride=1, padding=0)
        self.kpts_conv_q = nn.Conv2d(self.kpts_num, self.kpts_num, kernel_size=(1, 1), stride=1, padding=0)
        self.kpts_conv_v = nn.Conv2d(self.kpts_num, self.kpts_num, kernel_size=(1, 1), stride=1, padding=0)
        
        self.norm_fact = self.fea_dim

    def forward(self, kpt_feat, imgid, j, type=None):
        p, c, h, w = kpt_feat.size()
        # kpt_feat = self.joint_conv(kpt_feat)
        residual_kpt_feat = kpt_feat

        kpts_feat_k = self.kpts_conv_k(kpt_feat).view(p, self.kpts_num, -1).contiguous()    # [p, kpts, dim]
        kpts_feat_q = self.kpts_conv_q(kpt_feat).view(p, self.kpts_num, -1).contiguous()    # [p, kpts, dim]
        kpts_feat_v = self.kpts_conv_v(kpt_feat).view(p, self.kpts_num, -1).contiguous()    # [p, kpts, dim]

        imgid = imgid.cpu().numpy().tolist()
        dic, kpt_feats = {}, None
        
        for key in imgid:
            dic[key] = dic.get(key, 0) + 1
        start = 0
        dic_idx = list(dic.keys())
        for i in dic.keys():
            if start+dic[i] > len(imgid):
                break
            if dic_idx.index(i) == 0:
                kpt_k = kpts_feat_k[:dic[i], :, :].contiguous().view(-1, self.kpts_num, h * w)
                kpt_q = kpts_feat_q[:dic[i], :, :].contiguous().view(-1, self.kpts_num, h * w)
                kpt_v = kpts_feat_v[:dic[i], :, :].contiguous().view(-1, self.kpts_num, h * w)
                start = dic[i]
            else:
                kpt_k = kpts_feat_k[start:start+dic[i], :, :].contiguous().view(-1, self.kpts_num, h * w)
                kpt_q = kpts_feat_q[start:start+dic[i], :, :].contiguous().view(-1, self.kpts_num, h * w)
                kpt_v = kpts_feat_v[start:start+dic[i], :, :].contiguous().view(-1, self.kpts_num, h * w)
                start = start + dic[i]
            kpt_relation = torch.matmul(kpt_q, kpt_k.permute(0, 2, 1).contiguous()) / self.norm_fact   # [kpts, kpts]
            # if type == 'ij':
            #     np.save('/data2_12t/user/dyh/results/DHRNet/joint_relation/ij_joint_{}_{}.npy'.format(str(j), i), kpt_relation.cpu().detach().numpy())
            # else:
            #     np.save('/data2_12t/user/dyh/results/DHRNet/joint_relation/ji_joint_{}_{}.npy'.format(str(j), i), kpt_relation.cpu().detach().numpy())
            
            kpt_relation = F.softmax(kpt_relation, dim=0)
            # np.save('vis/res/correlation/joint_corr_att.npy', kpt_relation.cpu().detach().numpy())
            kpt_relation_feat = torch.matmul(kpt_relation, kpt_v).view(-1, self.kpts_num, h, w).contiguous()
            
            if dic_idx.index(i) == 0:
                kpt_feats = kpt_relation_feat
            else:
                kpt_feats = torch.cat([kpt_feats, kpt_relation_feat], dim=0)
        kpt_feats += residual_kpt_feat     # [p, k, h, w]
        
        return F.relu(kpt_feats)
        
        
class DualRelationModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self.inst_input_channels = cfg.MODEL.GFD.CHANNELS

        self.inst_feat_dim = cfg.DATASET.OUTPUT_SIZE
        self.kpts_num = cfg.DATASET.NUM_KEYPOINTS

        self.instance_relation = InstanceRelationModule(cfg, self.inst_input_channels)
        self.kpts_relation = JointRelationModule(cfg)

        self.instance_relation_2 = InstanceRelationModule(cfg, self.inst_input_channels)
        self.kpts_relation_2 = JointRelationModule(cfg)

        # SE
        self.se_block1 = SEBlock('avg', channels=self.inst_input_channels+self.kpts_num, ratio=16)
        self.se_block2 = SEBlock('avg', channels=self.inst_input_channels+self.kpts_num, ratio=16)

        # CBAM
        self.ca = ChannelAttention(self.kpts_num+self.inst_input_channels)
        self.sa = SpatialAttention()

        self.inst_kpt_conv = nn.Conv2d(self.inst_input_channels+self.kpts_num, self.kpts_num, 1, 1, 0)
        self.kpt_inst_conv = nn.Conv2d(self.kpts_num+self.inst_input_channels, self.inst_input_channels, 1, 1, 0)
        
        self.feat_fusion_conv = nn.Conv2d(self.kpts_num + self.inst_input_channels, self.inst_input_channels, 1, 1, 0)
        # self._init_weight()

    def forward(self, instance_feats, kpt_feats, imgid, instance_param, i):
        # instance_feats: [p, 32, 128, 128], kpt_feats: [p, kpts, 128, 128]
        
        # Instance-Joint order
        inst_relation_feat = self.instance_relation(instance_feats, imgid, instance_param, i, type='ij')
        inst_kpt_feat = torch.cat([inst_relation_feat, kpt_feats], dim=1)
        inst_kpt_feat = self.se_block1(inst_kpt_feat)
        inst_kpt_feat = self.inst_kpt_conv(inst_kpt_feat)
        kpt_relation_feat_1 = self.kpts_relation(inst_kpt_feat, imgid, i, type='ij')

        # Joint-Instance order
        kpt_relation_feat = self.kpts_relation_2(kpt_feats, imgid, i, type='ji')
        kpt_inst_feat = torch.cat([kpt_relation_feat, instance_feats], dim=1)
        kpt_inst_feat = self.se_block2(kpt_inst_feat)
        kpt_inst_feat = self.kpt_inst_conv(kpt_inst_feat)
        inst_relation_feat_2 = self.instance_relation_2(kpt_inst_feat, imgid, instance_param, i, type='ji')

        
        final_feats = torch.cat([kpt_relation_feat_1, inst_relation_feat_2], dim=1)
        final_feats = self.ca(final_feats) * final_feats
        final_feats = self.sa(final_feats) * final_feats
        final_feats = self.feat_fusion_conv(final_feats)
        
        return F.relu(final_feats + instance_feats)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)