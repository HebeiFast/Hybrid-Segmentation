import typing as t

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
# from ops.modules import MSDeformAttn
from ervit.modeling.pixel_decoder.msdeformattn import MSDeformAttn
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from timm.models.layers import DropPath

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate


_logger = logging.getLogger(__name__)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

NORM_EPS = 1e-5


class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = neuron.LIFNode(tau=2.,
                                  decay_input=True,
                                  v_threshold=1.,
                                  v_reset=0.,
                                  surrogate_function=surrogate.Sigmoid(),
                                  detach_reset=True,
                                  step_mode='m',
                                  backend='cupy')

    def forward(self, input):
        if isinstance(input, tuple):
            x, v = input
        else:
            x = input
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = self.norm(x)
        x = x.reshape(T, B, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        x = self.act(x)
        x = x.permute(1, 0, 2, 3, 4)
        v = self.act.v
        return x, v


class Adaptor(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.downsample = nn.Linear(dim, hidden)
        self.activation = nn.LeakyReLU(inplace=True)
        self.upsample = nn.Linear(hidden, dim)

    def forward(self, input):
        x = self.downsample(input)
        x = self.activation(x)
        x = self.upsample(x)
        return x

class TemporalWeighting(nn.Module):
    def __init__(self, T, C):
        super(TemporalWeighting, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptor = Adaptor(T*C, (T*C)//(C//2))
        self.softmax = nn.Softmax(dim=1)
  
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_pooled = self.global_avg_pool(x.view(B, T * C, H, W))
        x_pooled = x_pooled.reshape(B, 1, T * C)
        x_recon = self.adaptor(x_pooled).reshape(B, T, C)
 
        attn_weights = self.softmax(x_recon)

        weighted_x = (x * attn_weights.view(B, T, C, 1, 1)).sum(dim=1) / T 
        return weighted_x 

class E2I_Injector(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1,
                with_cffn=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()

        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points)

        self.with_cffn = with_cffn
        self.with_cp = with_cp
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        N_steps = dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

    def get_reference_point_sample_from_event(self, feat_e, downsample_e):
        B, C, H, W = feat_e.shape
        device = feat_e.device
        
        feat_e_sel_list = []
        feat_e_sel_length = []
        indices_list = []
        reference_point_list = []
        max_nonzero_num = 0
        
        for idx in range(B):
            non_zero_indices = torch.nonzero(downsample_e[idx], as_tuple=False)
            height_indices = non_zero_indices[:, 0]
            width_indices = non_zero_indices[:, 1]

            feat_e_sel = feat_e[idx, :, height_indices, width_indices]
            indices_list.append(non_zero_indices)
            feat_e_sel_list.append(feat_e_sel)
            
            non_zero_indices = non_zero_indices.float()
            non_zero_indices[:, 0] /= H
            non_zero_indices[:, 1] /= W
            reference_point_list.append(non_zero_indices)

            max_nonzero_num = max(max_nonzero_num, non_zero_indices.shape[0])
            feat_e_sel_length.append(non_zero_indices.shape[0])
        
        for idx in range(len(reference_point_list)):
            reference_points = reference_point_list[idx]
            feat_e_sel = feat_e_sel_list[idx]
            
            if reference_points.shape[0] < max_nonzero_num:
                sample_nums = max_nonzero_num - reference_points.shape[0]
                mask_points = -1 * torch.ones((sample_nums, 2), device=device)
                mask_feat_e = torch.zeros((C, sample_nums), device=device)
                reference_point_list[idx] = torch.cat((reference_points, mask_points), dim=0)
                feat_e_sel_list[idx] = torch.cat((feat_e_sel, mask_feat_e), dim=1)
        
        feat_e_sels = torch.stack(feat_e_sel_list, dim=0).transpose(1, 2)
        reference_points = torch.stack(reference_point_list, dim=0)
        reference_points = reference_points[:, :, None]
        
        return feat_e_sels, reference_points, indices_list, feat_e_sel_length

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, feat_i, feat_e, downsample_e):

        B, C, H, W = feat_i.shape

        spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=feat_i.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_i_pos2d = self.pe_layer(feat_i)
        feat_e_pos2d = self.pe_layer(feat_e)

        feat_i = self.with_pos_embed(feat_i, feat_i_pos2d)
        feat_e = self.with_pos_embed(feat_e, feat_e_pos2d)

    
        feat_e_sels, reference_points, indices_list, feat_e_sel_length = self.get_reference_point_sample_from_event(feat_e, downsample_e)
        feat_i = feat_i.flatten(2).transpose(1, 2)

        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)

            query = self.gamma * attn
 
            query_project = torch.zeros((B, H, W, C), device=feat_i.device)
            for idx in range(feat_e.shape[0]):
                indice = indices_list[idx]
                query_batch = query[idx, :feat_e_sel_length[idx]]
                query_project[idx, indice[:, 0], indice[:, 1], :] = query_batch

            return query_project

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, feat_e_sels, feat_i)
        else:
            query = _inner_forward(feat_e_sels, feat_i)

        query = query.transpose(1, 2).reshape(B, C, H, W)
        feat_i = feat_i.transpose(1, 2).reshape(B, C, H, W)

        query = feat_i + feat_e + query

        return query

class I2E_Injector(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1,
                with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):

        super().__init__()
        self.temporal_weighting = TemporalWeighting(T=5, C=dim)
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        N_steps = dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, feat_i, feat_e):
        feat_e = self.temporal_weighting(feat_e)
        B, C, H, W = feat_i.shape
        spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=feat_i.device)
        reference_image_points = get_reference_points([(H, W)], feat_i.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_i_pos = self.pe_layer(feat_i).flatten(2).transpose(1,2)
        feat_e_pos = self.pe_layer(feat_e).flatten(2).transpose(1,2)

        feat_i_flatten = feat_i.flatten(2).transpose(1,2)
        feat_e_flatten = feat_e.flatten(2).transpose(1,2)

        feat_i_flatten = self.with_pos_embed(feat_i_flatten, feat_i_pos)
        feat_e_flatten = self.with_pos_embed(feat_e_flatten, feat_e_pos)

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_image_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, feat_i_flatten, feat_e_flatten)
        else:
            query = _inner_forward(feat_i_flatten, feat_e_flatten)

        query = query.transpose(1, 2).reshape(B, C, H, W)
        return query

class Channel_Selection(nn.Module):
    def __init__(self, k_size=3):
        super(Channel_Selection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        select = x * y.expand_as(x)
        output = x + select

        return output

class InteractionModule_Deform(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1,
                with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, mode='I2E', fusion='add'):
        super().__init__()


        self.mode = mode
        self.fusion = fusion

        self.I2E_Inject = I2E_Injector(dim, num_heads=num_heads, n_points=n_points, n_levels=n_levels,
                with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                norm_layer=norm_layer, with_cp=with_cp)
        

        self.E2I_Inject = E2I_Injector(dim, num_heads=num_heads // 2, n_points=n_points, n_levels=n_levels,
                with_cffn=with_cffn,norm_layer=norm_layer, init_values=init_values, with_cp=with_cp)
        
        self.rgb_select = Channel_Selection()
        self.event_select = Channel_Selection()

    def forward(self, feat_i, feat_e, downsample_e):
        T = feat_e.shape[1]
        
        image_query = self.I2E_Inject(feat_i, feat_e)

        event_query_list = []
        for t in range(T):
            event_query = self.E2I_Inject(feat_i, feat_e[:, t, ...], downsample_e[:, t, ...])
            event_query_list.append(event_query)
        event_query_stack = torch.stack(event_query_list, dim=1)

        average_event = torch.mean(event_query_stack, 1)
        select_rgb = self.rgb_select(image_query)
        select_event = self.event_select(average_event)
        fusion = select_rgb + select_event
      
        return image_query, event_query_stack, fusion