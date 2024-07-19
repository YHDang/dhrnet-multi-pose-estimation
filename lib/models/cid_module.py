import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math
import pdb

from .loss import FocalLoss, ContrastiveLoss
# from .modules.dual_relation_ablation import *
from .modules.dual_relation import *

def build_iia_module(cfg):
    return IIA(cfg)

def build_gfd_module(cfg):
    return GFD(cfg)

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class IIA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.IIA.IN_CHANNELS    # 480
        self.out_channels = cfg.MODEL.IIA.OUT_CHANNELS  # num_keypoints + 1
        assert self.out_channels == self.num_keypoints + 1
        self.prior_prob = cfg.MODEL.BIAS_PROB   # 0.01
        self.DHRNet = cfg.DHRNET

        self.keypoint_center_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        torch.nn.init.normal_(self.keypoint_center_conv.weight, std=0.001)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.keypoint_center_conv.bias, bias_value)

        self.heatmap_loss = FocalLoss()
        self.contrastive_loss = ContrastiveLoss()

        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2

    def forward(self, features, j, batch_inputs=None):
        pred_multi_heatmap = _sigmoid(self.keypoint_center_conv(features)) # [batch, kpts+1, 128, 224]
        

        if self.training:
            gt_multi_heatmap = [x['multi_heatmap'].unsqueeze(0).to(self.device) for x in batch_inputs]  # Load GT multi heatmap
            gt_multi_heatmap = torch.cat(gt_multi_heatmap, dim=0)   # [batch, kpts+1, 128, 128]
            gt_multi_mask = [x['multi_mask'].unsqueeze(0).to(self.device) for x in batch_inputs]    # Load GT mask
            gt_multi_mask = torch.cat(gt_multi_mask, dim=0) # [batch, 1, 128, 128]

            multi_heatmap_loss = self.heatmap_loss(pred_multi_heatmap, gt_multi_heatmap, gt_multi_mask)

            contrastive_loss = 0
            total_instances = 0
            instances = defaultdict(list)

            for i in range(features.size(0)):
                if 'instance_coord' not in batch_inputs[i]: continue
                instance_coord = batch_inputs[i]['instance_coord'].to(self.device)  # [persons, 2]
                instance_heatmap = batch_inputs[i]['instance_heatmap'].to(self.device)  # [persons, kpts, 128, 128]
                instance_mask = batch_inputs[i]['instance_mask'].to(self.device)    # [persons, kpts, 1, 1]
                instance_imgid = i * torch.ones(instance_coord.size(0), dtype=torch.long).to(self.device)
                instance_param = self._sample_feats(features[i], instance_coord)    # [persons, channels] save all persons in each batch
                contrastive_loss += self.contrastive_loss(instance_param)
                total_instances += instance_coord.size(0)

                instances['instance_coord'].append(instance_coord)
                instances['instance_imgid'].append(instance_imgid)
                instances['instance_param'].append(instance_param)
                instances['instance_heatmap'].append(instance_heatmap)
                instances['instance_mask'].append(instance_mask)
            for k, v in instances.items():
                instances[k] = torch.cat(v, dim=0)
            return multi_heatmap_loss, contrastive_loss/total_instances, instances
        else:
            instances = {}
            W = pred_multi_heatmap.size()[-1]   # 224
            if self.flip_test:
                center_heatmap = pred_multi_heatmap[:, -1, :, :].mean(dim=0, keepdim=True)
            else:
                center_heatmap = pred_multi_heatmap[:, -1, :, :]    # [batch, h, w]

            # center_heatmap.shape: [batch, 128, 224], self.center_pool_kernel=3
            center_pool = F.avg_pool2d(center_heatmap, self.center_pool_kernel, 1, (self.center_pool_kernel-1)//2)  # [batch, 128, 224]
            center_heatmap = (center_heatmap + center_pool) / 2.0
            maxm = self.hierarchical_pool(center_heatmap)   # [batch, 128, 224]
            maxm = torch.eq(maxm, center_heatmap).float()   # 相等为1，不相等为0, [batch, 128, 224]
            center_heatmap = center_heatmap * maxm  # [batch, 128, 224]

            scores = center_heatmap.view(-1)    # [batch * 128 * 224]
            # self.max_proposals=30
            # Select top 30 proposals
            scores, pos_ind = scores.topk(self.max_proposals, dim=0)
            # self.keypoint_thre: 0.01
            select_ind = (scores > (self.keypoint_thre)).nonzero()  # [n, 1]: 大于keypoint_thre的点
            if len(select_ind) > 0:
                scores = scores[select_ind].squeeze(1) # [n]: 大于keypoint_thre的点的置信度
                pos_ind = pos_ind[select_ind].squeeze(1)    # [n]: 大于keypoint_thre的点的index
                # 将instance proposals的index转换为坐标
                x = pos_ind % W
                y = (pos_ind / W).long()
                instance_coord = torch.stack((y, x), dim=1) #[n, 2]
                # 从Image Feature中提取可能的Instance表征
                instance_param = self._sample_feats(features[0], instance_coord)
                instance_imgid = torch.zeros(instance_coord.size(0), dtype=torch.long).to(features.device) # [n]
                if self.flip_test:
                    instance_param_flip = self._sample_feats(features[1], instance_coord)
                    instance_imgid_flip = torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                    instance_coord = torch.cat((instance_coord, instance_coord), dim=0)
                    instance_param = torch.cat((instance_param, instance_param_flip), dim=0)
                    instance_imgid = torch.cat((instance_imgid, instance_imgid_flip), dim=0)

                instances['instance_coord'] = instance_coord
                instances['instance_imgid'] = instance_imgid
                instances['instance_param'] = instance_param
                instances['instance_score'] = scores
            return instances
    
    def _sample_feats(self, features, pos_ind):
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0)

    def hierarchical_pool(self, heatmap):
        # heatmap: [1, 128, 224]
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > self.pool_thre1:  # self.pool_thre1: 300
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > self.pool_thre2:    # self.pool_thre2: 200
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm # [1, 128, 224]

class GFD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.is_DHRNet=cfg.DHRNET
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.GFD.IN_CHANNELS
        self.channels = cfg.MODEL.GFD.CHANNELS
        self.out_channels = cfg.MODEL.GFD.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints
        self.prior_prob = cfg.MODEL.BIAS_PROB

        self.conv_down = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.c_attn = ChannelAtten(self.in_channels, self.channels)
        self.s_attn = SpatialAtten(self.in_channels, self.channels)
        self.fuse_attn = nn.Conv2d(self.channels*2, self.channels, 1, 1, 0)
        self.heatmap_conv = nn.Conv2d(self.channels, self.out_channels, 1, 1, 0)

        if self.is_DHRNet:
            self.pre_heatmap_conv = nn.Conv2d(self.channels, self.out_channels, 1, 1, 0)
            self.dual_relation = DualRelationModule(cfg)
        self.heatmap_loss = FocalLoss()

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
        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.heatmap_conv.bias.data.fill_(bias_value)

    def forward(self, features, instances, i):
        # features: [batch, 480, 128, 128]
        global_features = self.conv_down(features)  # [batch, 32, 128, 128]
        instance_features = global_features[instances['instance_imgid']]    #[p, 32, 128, 128], instance_imgid: save persons in each image
        instance_params = instances['instance_param']   # [p, 480] p is the number of instance proposals
        c_instance_feats = self.c_attn(instance_features, instance_params)  #[p, 32, 128, 128]
        s_instance_feats = self.s_attn(instance_features, instance_params, instances['instance_coord']) # [p, 32, 128, 128]
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats), dim=1)    # [p, 32*2, 128, 128]
        cond_instance_feats = self.fuse_attn(cond_instance_feats)   # [p, 32, 128, 128]
        cond_instance_feats = F.relu(cond_instance_feats)   # [p, 32, 128, 128]

        if self.is_DHRNet:
            mid_instance_heatmaps = F.relu(self.pre_heatmap_conv(cond_instance_feats))   # [p, kpts, 128, 128]
            cond_instance_feats = self.dual_relation(cond_instance_feats, mid_instance_heatmaps, instances['instance_imgid'], instance_params, i)
        
        # pdb.set_trace()
        pred_instance_heatmaps = _sigmoid(self.heatmap_conv(cond_instance_feats))
        

        if self.training:
            gt_instance_heatmaps = instances['instance_heatmap']
            gt_instance_masks = instances['instance_mask']  # [p, kpts, 1, 1]
            single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps, gt_instance_heatmaps, gt_instance_masks)
            return single_heatmap_loss
        else:
            return pred_instance_heatmaps

class ChannelAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels) # [32, 32]

    def forward(self, global_features, instance_params):
        # input_size: [b, 32, 128, 128], [b, 480]
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1) # [b, 32, 1, 1]
        return global_features * instance_params.expand_as(global_features)

class SpatialAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels) # [32, 32]
        self.feat_stride = 4
        conv_in = 3
        self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2)

    def forward(self, global_features, instance_params, instance_inds):
        # global_features: [b, 32, 128, 128], instance_params: [b, 480], instance_inds: [b, 2]
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1) # [b, 32, 1, 1], Compute channel features of each instance
        feats = global_features * instance_params.expand_as(global_features)    # [b, 32, 128, 128], fuse channel attention with original features
        fsum = torch.sum(feats, dim=1, keepdim=True)    # [b, 1, 128, 128]
        input_feats = fsum
        locations = compute_locations(global_features.size(2), global_features.size(3), stride=1, device=global_features.device)    # [size(2) * size(3), 2]
        n_inst = instance_inds.size(0)  # 54 -> b
        H, W = global_features.size()[2:]
        instance_locations = torch.flip(instance_inds, [1]) # [b, 2]
        instance_locations = instance_locations
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)  
        relative_coords = relative_coords.permute(0, 2, 1).float()  # [b, 2, 128*128]
        relative_coords = (relative_coords / 32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)  # [b, 2, 128, 128]
        input_feats = torch.cat((input_feats, relative_coords), dim=1)  # [b, 3, 128, 128]
        mask = self.conv(input_feats).sigmoid()
        return global_features * mask

def compute_locations(h, w, stride, device):    # h: 128, w: 128, stride: 1
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)   # shift_y: [128, 128]; shift_x: [128, 128]
    shift_x = shift_x.reshape(-1)   # [16384] -> [128*128] 
    shift_y = shift_y.reshape(-1)   # [16384]
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2    # [16384, 2]
    return locations