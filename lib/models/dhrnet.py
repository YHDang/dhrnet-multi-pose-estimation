import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import logging

from .backbone import build_backbone
from .cid_module import build_iia_module, build_gfd_module


logger = logging.getLogger(__name__)

class DHRNet(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg, is_train)
        self.iia = build_iia_module(cfg)
        self.gfd = build_gfd_module(cfg)

        self.multi_heatmap_loss_weight = cfg.LOSS.MULTI_HEATMAP_LOSS_WEIGHT
        self.contrastive_loss_weight = cfg.LOSS.CONTRASTIVE_LOSS_WEIGHT
        self.single_heatmap_loss_weight = cfg.LOSS.SINGLE_HEATMAP_LOSS_WEIGHT

        # inference
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_index = cfg.DATASET.FLIP_INDEX
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2
    
    def forward(self, batch_inputs, i):
        images = [x['image'].unsqueeze(0).to(self.device) for x in batch_inputs]
        # images = [batch_inputs['image'].unsqueeze(0).to(self.device)]
        images = torch.cat(images, dim=0)
        
        ## Save Input Image
        np.save('/data2_12t/user/dyh/results/DHRNet/input/input_{}.npy'.format(str(i)), images.detach().cpu().numpy())
        feats = self.backbone(images) # [batch, 480, 128, 128]

        if self.training:
            multi_heatmap_loss, contrastive_loss, instances = self.iia(feats, i, batch_inputs)
            # limit max instances in training
            if 0 <= self.max_instances < instances['instance_param'].size(0):
                inds = torch.randperm(instances['instance_param'].size(0), device=self.device).long()
                for k, v in instances.items():
                    instances[k] = v[inds[:self.max_instances]]

            single_heatmap_loss = self.gfd(feats, instances)

            losses = {}
            losses.update({'multi_heatmap_loss': multi_heatmap_loss * self.multi_heatmap_loss_weight})
            losses.update({'single_heatmap_loss': single_heatmap_loss * self.single_heatmap_loss_weight})
            losses.update({'contrastive_loss': contrastive_loss * self.contrastive_loss_weight})
            return losses
        else:
            results = {}
            if self.flip_test:
                feats[1, :, :, :] = feats[1, :, :, :].flip([2])

            instances = self.iia(feats, i)

            if len(instances) == 0: return results

            instance_heatmaps = self.gfd(feats, instances, i)

            if self.flip_test:
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(instance_heatmaps, 2, dim=0)
                instance_heatmaps_flip = instance_heatmaps_flip[:, self.flip_index, :, :]
                instance_heatmaps = (instance_heatmaps + instance_heatmaps_flip) / 2.0

            instance_scores = instances['instance_score']
            num_people, num_keypoints, h, w = instance_heatmaps.size()
            center_pool = F.avg_pool2d(instance_heatmaps, self.center_pool_kernel, 1, (self.center_pool_kernel-1)//2)
            instance_heatmaps = (instance_heatmaps + center_pool) / 2.0
            
            nms_instance_heatmaps = instance_heatmaps.view(num_people, num_keypoints, -1)
            
            vals, inds = torch.max(nms_instance_heatmaps, dim=2)
            x, y = inds % w, (inds / w).long()
            # shift coords by 0.25
            x, y = self.adjust(x, y, instance_heatmaps)
            
            vals = vals * instance_scores.unsqueeze(1)
            poses = torch.stack((x, y, vals), dim=2)

            poses[:, :, :2] = poses[:, :, :2] * 4 + 2
            scores = torch.mean(poses[:, :, 2], dim=1)

            results.update({'poses': poses})
            results.update({'scores': scores})

            return results

    def adjust(self, res_x, res_y, heatmaps):
        n, k, h, w = heatmaps.size()#[2:]

        # Create the range of coordinations
        x_l, x_r = (res_x - 1).clamp(min=0), (res_x + 1).clamp(max=w-1)
        y_t, y_b = (res_y + 1).clamp(max=h-1), (res_y - 1).clamp(min=0)
        n_inds = torch.arange(n)[:, None].to(self.device)
        k_inds = torch.arange(k)[None].to(self.device)

        px = torch.sign(heatmaps[n_inds, k_inds, res_y, x_r] - heatmaps[n_inds, k_inds, res_y, x_l])*0.25
        py = torch.sign(heatmaps[n_inds, k_inds, y_t, res_x] - heatmaps[n_inds, k_inds, y_b, res_x])*0.25

        res_x, res_y = res_x.float(), res_y.float()
        x_l, x_r = x_l.float(), x_r.float()
        y_b, y_t = y_b.float(), y_t.float()
        px = px*torch.sign(res_x-x_l)*torch.sign(x_r-res_x)
        py = py*torch.sign(res_y-y_b)*torch.sign(y_t-res_y)

        res_x = res_x.float() + px
        res_y = res_y.float() + py

        return res_x, res_y

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                # if name.split('.')[0] in self.pretrained_layers \
                #    or self.pretrained_layers[0] is '*':
                if name in parameters_names or name in buffers_names:
                    if verbose:
                        logger.info(
                            '=> init {} from {}'.format(name, pretrained)
                        )
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)