# Copyright (c) ByteDance Inc. All rights reserved.
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from .evit import Spiking_EViT, Spiking_NCB, Spiking_NTB
from .rvit import RViT, NCB, NTB

from .adapter_modules import InteractionModule_Deform
from ervit.modeling.pixel_decoder.msdeformattn import MSDeformAttn

class ERViT(nn.Module):
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False, resume='', with_extra_norm=True, frozen_stages=-1,
                 norm_eval=False, norm_cfg=None, mode='I2E', fusion='fusion',
                 ):
        super(ERViT, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.frozen_stages = frozen_stages
        self.with_extra_norm = with_extra_norm
        self.norm_eval = norm_eval

        self.RViT = RViT(stem_chs, depths, path_dropout, attn_drop, drop, num_classes,
                 strides, sr_ratios, head_dim, mix_block_ratio,
                 use_checkpoint, resume, with_extra_norm, frozen_stages,
                 norm_eval, norm_cfg)
        self.Spiking_EViT = Spiking_EViT(stem_chs, depths, path_dropout, attn_drop, drop, num_classes,
                 strides, sr_ratios, head_dim, mix_block_ratio,
                 use_checkpoint, resume, with_extra_norm, frozen_stages,
                 norm_eval, norm_cfg)
        
        self.num_features = [32, 64, 128, 128]
        self.interaction_modules = []
        
        for idx, num_feature in enumerate(self.num_features):
            self.interaction_modules.append(InteractionModule_Deform(num_feature, mode=mode, fusion=fusion))
        self.interaction_modules = nn.Sequential(*self.interaction_modules)

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]

        print('initialize_weights...')
        self._initialize_weights()
        if resume:
            self.init_weights(resume)
        if norm_cfg is not None:
            self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ERViT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, NCB) or isinstance(module, NTB) or isinstance(module, Spiking_NCB) or isinstance(module, Spiking_NTB):
                module.merge_bn()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print('\n using pretrained model\n')
            checkpoint = torch.load(pretrained, map_location='cpu')['model']
            self.load_state_dict(checkpoint, strict=False)

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)            
            elif isinstance(m, MSDeformAttn):
                m._reset_parameters()


    def forward(self, image, event):
        outputs = {}

        x_i = self.RViT.image_stem(image)
        x_e = self.Spiking_EViT.event_stem(event)

        event_merged = torch.sum(event, dim=2) # B, T, H, W
        event_merged_scale = []
        downsample_factor = 0.25
        for i in range(len(self.num_features)):
            downsample_event = F.interpolate(event_merged, scale_factor=downsample_factor, mode='bilinear', align_corners=False)
            event_merged_scale.append(downsample_event)
            downsample_factor = downsample_factor / 2

        stage_id = 0
        for idx, (image_layer, event_layer) in enumerate(zip(self.RViT.features, self.Spiking_EViT.features)):
            if self.use_checkpoint:
                x_i = checkpoint.checkpoint(image_layer, x_i)
                x_e = checkpoint.checkpoint(event_layer, x_e)
            else:
                x_i = image_layer(x_i)
                x_e = event_layer(x_e)

            if idx == self.stage_out_idx[stage_id]:
                if self.with_extra_norm:
                    B, T, C, H, W = x_e.shape
                    x_e = x_e.flatten(0, 1)
                    if stage_id < 3:
                        x_i = self.RViT.extra_norm_list[stage_id](x_i)
                        x_e = self.Spiking_EViT.extra_norm_list[stage_id](x_e)
                    else:
                        x_i = self.RViT.norm(x_i)
                        x_e = self.Spiking_EViT.norm(x_e)

                x_e = x_e.reshape(B, T, x_e.shape[1], x_e.shape[2], x_e.shape[3]).contiguous()
                x_i, x_e, fusion_ie = self.interaction_modules[stage_id](x_i, x_e, event_merged_scale[stage_id])

                outputs["res{}".format(stage_id + 2)] = fusion_ie
                stage_id += 1
        return outputs


@BACKBONE_REGISTRY.register()
class ERViT_Tiny(ERViT, Backbone):
    def __init__(self, cfg, input_shape):

        stem_chs = cfg.MODEL.ERVIT.STEM_CHS
        depths = cfg.MODEL.ERVIT.DEPTHS
        path_dropout = cfg.MODEL.ERVIT.PATH_DROPOUT
        attn_drop = cfg.MODEL.ERVIT.ATTN_DROP
        drop = cfg.MODEL.ERVIT.DROP
        strides = cfg.MODEL.ERVIT.STRIDES
        sr_ratios = cfg.MODEL.ERVIT.SR_RATIOS
        head_dim = cfg.MODEL.ERVIT.HEAD_DIM
        mix_block_ratio = cfg.MODEL.ERVIT.MIX_BLOCK_RATIO
        use_checkpoint = cfg.MODEL.ERVIT.USE_CHECKPOINT
        resume = cfg.MODEL.ERVIT.RESUME
        with_extra_norm = cfg.MODEL.ERVIT.WITH_EXTRA_NORM
        frozen_stages = cfg.MODEL.ERVIT.FROZEN_STAGES
        norm_eval = cfg.MODEL.ERVIT.NORM_EVAL
        norm_cfg = cfg.MODEL.ERVIT.NORM_CFG
        mode = cfg.MODEL.ERVIT.MODE
        fusion = cfg.MODEL.ERVIT.FUSION

        super(ERViT_Tiny, self).__init__(
            stem_chs = stem_chs,
            depths = depths,
            path_dropout = path_dropout,
            attn_drop = attn_drop,
            strides = strides,
            sr_ratios = sr_ratios,
            head_dim = head_dim,
            mix_block_ratio = mix_block_ratio,
            use_checkpoint = use_checkpoint,
            resume = resume,
            with_extra_norm = with_extra_norm,
            frozen_stages = frozen_stages,
            norm_eval = norm_eval,
            norm_cfg = norm_cfg,
            mode = mode,
            fusion = fusion
        )

        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, image, event):
        outputs = {}
        y = super().forward(image, event)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32