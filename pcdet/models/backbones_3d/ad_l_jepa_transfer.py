# AD-L-JEPA for transferring pretraining from waymo to kitti
# the only difference is set conv_input's input channel to 4 instead of 5

from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block



class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Encoder(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.masked_ratio = model_cfg.MASKED_RATIO
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(4, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                # [200, 175, 5] -> [200, 176, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None



    def forward(self, batch_dict, voxel_features_partial=None, voxel_coords_partial=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded voxel feature: sparse tensor (B, 128, 2, 200, 176)
        """
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)   
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        feats = out.dense()
        return feats


        
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        feats = out
        return feats

class AD_L_JEPA_Transfer(nn.Module):
    """
    pre-trained model
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.masked_ratio = model_cfg.MASKED_RATIO
        
        self.num_point_features = 16 # for building the model

        self.encoder = Encoder(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False


        self.predictor = Predictor()

        # dict to save results, used for visualization or debugging
        self.forward_re_dict = {}

        down_factor = 8
        self.down_factor = down_factor
        self.grid = 1
        self.bev_x_shape = round((self.point_cloud_range[3] - self.point_cloud_range[0])/self.voxel_size[0]/self.down_factor)
        self.bev_y_shape = round((self.point_cloud_range[4] - self.point_cloud_range[1])/self.voxel_size[1]/self.down_factor)
        self.mask_token = nn.Parameter(torch.randn(1, 256, 1, 1), requires_grad=True)
        self.empty_token = nn.Parameter(torch.randn(1, 256, 1, 1), requires_grad=True)


    def get_loss(self, tb_dict=None):
        def reg_fn(z):
            return torch.sqrt(z.var(dim=0) + 1e-4)

        tb_dict = {} if tb_dict is None else tb_dict

        # read context, prediction, target
        context = self.forward_re_dict['context']
        prediction =  self.forward_re_dict['prediction']
        target = self.forward_re_dict['target']

        BATCH_SIZE = context.shape[0]

        # permute, such that feature dimension is the last dimension
        target = target.permute(0, 2, 3, 1).contiguous() # [B, 256, 200, 176] -> [B, 200, 176, 256]
        prediction = prediction.permute(0, 2, 3, 1).contiguous() # [B, 256, 200, 176] -> [B, 200, 176, 256]
        context = context.permute(0, 2, 3, 1).contiguous() # [B, 256, 200, 176] -> [B, 200, 176, 256]

        cos_sim_occ = F.cosine_similarity(target, prediction, dim=3) # [B, 200, 176]
        loss_occ_cos_jepa = 1.0 -cos_sim_occ
        
        # indices: 1 context foreground, 2 target foreground, 3 context empty, 4 target empty
        indices = torch.zeros(loss_occ_cos_jepa.shape).to(loss_occ_cos_jepa.device)
        indices[self.forward_re_dict['bev_mask_encoder']] = 1
        indices[self.forward_re_dict['bev_mask_target_encoder']] = 2
        indices[self.forward_re_dict['bev_mask_encoder_empty']] = 3
        indices[self.forward_re_dict['bev_mask_target_encoder_empty']] = 4

        context_context_voxels = context[indices==1] # [B, 200, 176, 256], [B, 200, 176] -> context_num x 256
        target_target_voxels = target[indices==2] # [B, 200, 176, 128], [B, 200, 176] -> target_num x 256
        prediction_target_voxels = prediction[indices==2] # [B, 200, 176, 128], [B, 200, 176] -> target_num x 256
        prediction_target_empty_voxels = prediction[indices==4] # [B, 200, 176, 128], [B, 200, 176] -> target_num x 256

        # variance for logs
        var_context_context_voxels = torch.var(context_context_voxels, dim=0).mean()
        var_target_target_voxels = torch.var(target_target_voxels, dim=0).mean()
        var_prediction_target_empty_voxels = torch.var(prediction_target_empty_voxels, dim=0).mean()
        var_prediction_target_voxels = torch.var(prediction_target_voxels, dim=0).mean()
        
        # loss for logs
        # cos jepa loss
        loss_cos_jepa_target_voxels = loss_occ_cos_jepa[indices==2]
        loss_cos_jepa_target_empty_voxels = loss_occ_cos_jepa[indices==4]
        loss_jepa = 0.75*loss_cos_jepa_target_voxels.mean() + 0.25*loss_cos_jepa_target_empty_voxels.mean()
        #loss_jepa = loss_occ_cos_jepa[(indices == 2) | (indices == 4)].mean()


        # reg loss
        context_context_voxels_over_samples = [context[b][(indices[b] == 1)].view(-1, 256) for b in range(BATCH_SIZE)]
        pstd_context_context_voxels_over_samples = [reg_fn(t) for t in context_context_voxels_over_samples]
        loss_reg_context_context_voxels_over_samples = [torch.mean(F.relu(1./16.-t)) for t in pstd_context_context_voxels_over_samples]
        loss_reg_context_context_voxels_over_samples = torch.stack(loss_reg_context_context_voxels_over_samples)
        #print("loss_reg_context_context_voxels_over_samples", loss_reg_context_context_voxels_over_samples)
        loss_reg_context_context_voxels = torch.mean(loss_reg_context_context_voxels_over_samples)
        
        target_target_voxels_over_samples = [target[b][(indices[b] == 2)].view(-1, 256) for b in range(BATCH_SIZE)]
        pstd_target_target_voxels_over_samples = [reg_fn(t) for t in target_target_voxels_over_samples]
        loss_reg_target_target_voxels_over_samples = [torch.mean(F.relu(1./16.-t)) for t in pstd_target_target_voxels_over_samples]
        loss_reg_target_target_voxels_over_samples = torch.stack(loss_reg_target_target_voxels_over_samples)
        loss_reg_target_target_voxels = torch.mean(loss_reg_target_target_voxels_over_samples)

        prediction_target_voxels_over_samples = [prediction[b][(indices[b] == 2)].view(-1, 256) for b in range(BATCH_SIZE)]       
        pstd_prediction_target_voxels_over_samples = [reg_fn(t) for t in prediction_target_voxels_over_samples]
        loss_reg_prediction_target_voxels_over_samples  = [torch.mean(F.relu(1./16.-t)) for t in pstd_prediction_target_voxels_over_samples]
        loss_reg_prediction_target_voxels_over_samples = torch.stack(loss_reg_prediction_target_voxels_over_samples)
        loss_reg_prediction_target_voxels = torch.mean(loss_reg_prediction_target_voxels_over_samples)
        
        loss_reg = loss_reg_prediction_target_voxels + loss_reg_context_context_voxels
    
        # overall loss
        loss = loss_jepa+10*loss_reg
        
        tb_dict = {
            'loss_jepa': loss_jepa.item(),
            'loss_reg': loss_reg.item(),
            'loss_reg_context_context_voxels': loss_reg_context_context_voxels.item(),
            'loss_reg_target_target_voxels': loss_reg_target_target_voxels.item(),
            'loss_reg_prediction_target_voxels': loss_reg_prediction_target_voxels.item(),
            'loss_cos_jepa_target_voxels': loss_cos_jepa_target_voxels.mean().item(),
            'loss_cos_jepa_target_empty_voxels': loss_cos_jepa_target_empty_voxels.mean().item(),
            'var_context_context_voxels': var_context_context_voxels.item(),
            'var_target_target_voxels': var_target_target_voxels.item(),
            'var_prediction_target_voxels': var_prediction_target_voxels.item(),
            'var_prediction_target_empty_voxels':var_prediction_target_empty_voxels.item()
        }

        return loss, tb_dict

    def get_voxel_feature(self):
        return self.forward_re_dict['context'], self.forward_re_dict['prediction'], self.forward_re_dict['target'], self.forward_re_dict['voxel_coords_partial'], self.forward_re_dict['unselect_voxel_coords_partial'], self.forward_re_dict['encoder_indices'], self.forward_re_dict['target_encoder_indices']

    def forward(self, batch_dict):
        voxel_features, coors = batch_dict['voxel_features'], batch_dict['voxel_coords']
        voxel_features = voxel_features[:, 0:4]

        ### down sample voxel features to bev feature size
        coor_down_sample = coors.detach().int().clone() # [points_num, 4], (batch_idx, z_idx, y_idx, x_idx)
        coor_down_sample[:, 1:] = torch.div(coor_down_sample[:, 1:], self.down_factor * self.grid, rounding_mode='floor')
        coor_down_sample[:, 1] = torch.div(coor_down_sample[:, 1], coor_down_sample[:, 1].max() * 2, rounding_mode='floor') 
        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0) # unique_coor_down_sample: [unique_points_num, 4], (batch_idx, z_idx, y_idx, x_idx); inverse_index: [points_num]
        
        ### mask on bev feature
        select_ratio = 1 - self.masked_ratio # ratio for select voxel
        nums = unique_coor_down_sample.shape[0]
        len_keep = int(nums * select_ratio)
        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)
        keep = ids_shuffle[:len_keep]
        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach()
        unique_keep_bool[keep] = 1
        bev_coords_encoder = unique_coor_down_sample[keep, :].long()
        bev_coords_target_encoder = unique_coor_down_sample[~unique_keep_bool.bool(), :].long()
        
        bev_mask_encoder = torch.zeros([batch_dict['batch_size'], self.bev_x_shape, self.bev_y_shape], dtype=torch.bool).to(voxel_features.device)
        bev_mask_encoder[bev_coords_encoder[:, 0], bev_coords_encoder[:, 2], bev_coords_encoder[:, 3]] = True
        bev_mask_target_encoder = torch.zeros([batch_dict['batch_size'], self.bev_x_shape, self.bev_y_shape], dtype=torch.bool).to(voxel_features.device)
        bev_mask_target_encoder[bev_coords_target_encoder[:, 0], bev_coords_target_encoder[:, 2], bev_coords_target_encoder[:, 3]] = True
        
        bev_mask_nonempty = torch.logical_or(bev_mask_encoder, bev_mask_target_encoder)
        bev_mask_empty = ~bev_mask_nonempty
        bev_mask_tmp = torch.rand((batch_dict['batch_size'], self.bev_x_shape, self.bev_y_shape), device=voxel_features.device) < (1-self.masked_ratio)
        bev_mask_encoder_empty = torch.logical_and(bev_mask_tmp, bev_mask_empty)
        bev_mask_target_encoder_empty = torch.logical_and(~bev_mask_tmp, bev_mask_empty)

        bev_mask_encoder_all = torch.logical_or(bev_mask_encoder, bev_mask_encoder_empty)

        ### upsample bev mask to point cloud
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index) # [points_num], boolean
        ids_keep = ids_keep.bool()
        ids_mask = ~ids_keep # [points_num], boolean
        
        ### mask input point cloud
        voxel_features_encoder, voxel_coords_encoder = voxel_features[ids_keep,:], coors[ids_keep,:]
        voxel_features_target_encoder, voxel_coords_target_encoder = voxel_features[ids_mask,:], coors[ids_mask,:]

        ### forward
        mask_token = self.mask_token.repeat(batch_dict['batch_size'], 1, self.bev_x_shape, self.bev_y_shape)
        empty_token = self.empty_token.repeat(batch_dict['batch_size'], 1, self.bev_x_shape, self.bev_y_shape)
       
        context = self.encoder(batch_dict, voxel_features_encoder, voxel_coords_encoder)
        bs, c, d, h, w = context.shape
        context = context.reshape(bs, -1, h, w)
        bev_mask_encoder_all = bev_mask_encoder_all.unsqueeze(1)
        context = torch.where(~bev_mask_encoder_all, mask_token, context)
        bev_mask_encoder_empty = bev_mask_encoder_empty.unsqueeze(1)
        context = torch.where(bev_mask_encoder_empty, empty_token, context)
        context = F.normalize(context, p=2, dim=1)
         
        prediction = self.predictor(context)
        prediction = F.normalize(prediction, p=2, dim=1)

        target = self.target_encoder(batch_dict, voxel_features_target_encoder, voxel_coords_target_encoder)
        bs, c, d, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        bev_mask_target_encoder_empty = bev_mask_target_encoder_empty.unsqueeze(1)
        target = torch.where(bev_mask_target_encoder_empty, empty_token, target)
        target = F.normalize(target, p=2, dim=1)
        
        self.forward_re_dict['context'] = context
        self.forward_re_dict['prediction'] = prediction
        self.forward_re_dict['target'] = target
        self.forward_re_dict['voxel_coords_partial'] = voxel_coords_encoder # (selected_size, 4) [batch_idx, z_idx, y_idx, x_idx]
        self.forward_re_dict['unselect_voxel_coords_partial'] = voxel_coords_target_encoder # (unselected_size, 4) [batch_idx, z_idx, y_idx, x_idx]
        self.forward_re_dict['encoder_indices'] = bev_coords_encoder
        self.forward_re_dict['target_encoder_indices'] = bev_coords_target_encoder
        self.forward_re_dict['bev_mask_encoder'] = bev_mask_encoder.squeeze(1)
        self.forward_re_dict['bev_mask_target_encoder'] = bev_mask_target_encoder.squeeze(1)
        self.forward_re_dict['bev_mask_encoder_empty'] = bev_mask_encoder_empty.squeeze(1)
        self.forward_re_dict['bev_mask_target_encoder_empty'] = bev_mask_target_encoder_empty.squeeze(1)   

        return batch_dict