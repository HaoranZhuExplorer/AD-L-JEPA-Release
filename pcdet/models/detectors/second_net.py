from .detector3d_template import Detector3DTemplate
from tools.visual_utils import evaluate_features_utils
import torch
import numpy as np
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict,  plot=False, final_output_dir=None):
        input_batch = batch_dict
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
    
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if plot:
                gt_boxes = input_batch['gt_boxes'] # [B, M, 8]
                num_classes = len(torch.unique(gt_boxes[:,:,7]))
                occ_feature = self.backbone_3d.get_voxel_feature()
                print("debug occ_feature", occ_feature.shape)
                bev_feature = occ_feature.reshape(occ_feature.shape[0], 256, 200, 176) # [B, 256, 200, 176]
                bev_feature = bev_feature.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                bev_feature = bev_feature.reshape(bev_feature.size(0), -1, bev_feature.size(3))  # shape: [batch_size, 200x176, 256] 
                #evaluate_features_utils.viz_cluster_bev_feature(bev_feature, final_output_dir+"/bev_{}.png".format(str(batch_dict['frame_id'])), 2) 
                #evaluate_features_utils.get_bev_map(input_batch['points'], final_output_dir+"/bev_{}_lidar_data.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_downsampled_lidar_data.png".format(str(batch_dict['frame_id'])))
                #self.bev_svd_analysis(bev_feature, final_output_dir+"/bev_{}_svd.png".format(str(batch_dict['frame_id'])))
                
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def extract_features(self, batch_dict, final_output_dir=None):
        # dataset settings
        distance_threshold=10
        voxel_size=(0.05, 0.05, 0.1)
        grid_size=(176, 200, 40)
        min_bound=(0, -40, -3)
        max_bound=(70.4, 40, 1)
        
        input_batch = batch_dict
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        occ_feature = self.backbone_3d.get_voxel_feature() # [B, 128, 2, 200, 176]
        points = input_batch['points'] # [num_points, 5], (batch_idx, x, y, z, intensity)
        #print("debug points", points.shape)
        points = points[:, 1:4] # [num_points, 3], (x, y, z)
        gt_boxes = input_batch['gt_boxes'][0] # [B, M, 8]
        min_bound = torch.tensor(min_bound, device=points.device, dtype=points.dtype)
        max_bound = torch.tensor(max_bound, device=points.device, dtype=points.dtype)
        voxel_size = torch.tensor(voxel_size, device=points.device, dtype=points.dtype)
        in_bounds_mask = (points >= min_bound).all(dim=1) & (points <= max_bound).all(dim=1)
        points = points[in_bounds_mask]
        downsample_ratios = torch.tensor([8.0, 8.0, 41.0/2], device=points.device, dtype=torch.float32)
        grid_indices = torch.floor((points - min_bound) / voxel_size / downsample_ratios).int()
        unique_grid_indices = torch.unique(grid_indices, dim=0) # [num_unique_voxels, 3], (x, y, z)
        feature_list = []
        label_list = []
        for voxel in unique_grid_indices:
            voxel_center = (voxel + torch.tensor([0.5, 0.5, 0.5]).to(points.device)) * downsample_ratios * voxel_size + min_bound
            intersect = False
            intersect_label = None
            for i in range(gt_boxes.shape[0]):
                gt_box_i = gt_boxes[i, 0:7]
                voxel_box = [[voxel_center[0], voxel_center[1], voxel_center[2], voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2], 0]]
                gt_box_i_tensor = gt_box_i.unsqueeze(0).to(points.device) 
                voxel_box_tensor = torch.tensor(voxel_box).to(points.device)
                distance = (gt_box_i_tensor[0:3]-voxel_box_tensor[0:3]).norm()
                if distance < distance_threshold:
                    iou = boxes_iou3d_gpu(gt_box_i_tensor, voxel_box_tensor)
                    if iou >0:
                        intersect = True
                        intersect_label = int(gt_boxes[i, 7])
                        break
            if intersect:
                feature = occ_feature[0, :, voxel[2], voxel[1], voxel[0]]
                label = intersect_label
                feature_list.append(feature)
                label_list.append(label)
                
        return feature_list, label_list

    # more than 10x faster than extract_features(). As it's difficult to vectorize the code over batches for this function, empirically bs=1 for each process works the best.
    def extract_features_faster(self, batch_dict, final_output_dir=None, distance_threshold=3.0, voxel_size=(0.05, 0.05, 0.1), min_bound=(0, -40, -3), max_bound=(70.4, 40, 1), downsample_ratios=(8.0, 8.0, 41.0/2)):
        
        min_bound = torch.tensor(min_bound, device=batch_dict['points'].device, dtype=batch_dict['points'].dtype)
        max_bound = torch.tensor(max_bound, device=batch_dict['points'].device, dtype=batch_dict['points'].dtype)
        voxel_size = torch.tensor(voxel_size, device=batch_dict['points'].device, dtype=batch_dict['points'].dtype)
        downsample_ratios = torch.tensor(downsample_ratios, device=batch_dict['points'].device, dtype=torch.float32)

        input_batch = batch_dict
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        occ_feature = self.backbone_3d.get_voxel_feature() # [B, 128, 2, 200, 176]

        feature_list = []
        label_list = []
        for b in range(occ_feature.shape[0]):
            points = input_batch['points'][input_batch['points'][:, 0]==b] # [num_points, 5], (batch_idx, x, y, z, intensity)
            points = points[:, 1:4] # [num_points, 3], (x, y, z)
            in_bounds_mask = (points >= min_bound).all(dim=1) & (points <= max_bound).all(dim=1)
            points = points[in_bounds_mask]
            grid_indices = torch.floor((points - min_bound) / voxel_size / downsample_ratios).int()
            unique_grid_indices = torch.unique(grid_indices, dim=0) # [num_unique_voxels, 3], (x, y, z)
            voxel_centers = (unique_grid_indices + torch.tensor([0.5, 0.5, 0.5]).to(points.device)) * downsample_ratios * voxel_size + min_bound
            
            gt_boxes =  input_batch['gt_boxes'][b] 
            
            for i in range(gt_boxes.shape[0]):
                gt_box_i = gt_boxes[i, 0:7]
                gt_box_i_tensor = gt_box_i.unsqueeze(0).to(points.device) 
                distance = torch.norm(gt_box_i[0:3]-voxel_centers[:, 0:3], dim=1)
                selected_voxel_indices = torch.where(distance < distance_threshold)
                selected_voxel_centers = voxel_centers[selected_voxel_indices]
                selected_voxels = unique_grid_indices[selected_voxel_indices]
                for j, voxel_center in enumerate(selected_voxel_centers):
                    voxel_box = [[voxel_center[0], voxel_center[1], voxel_center[2], voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2], 0]]
                    voxel_box_tensor = torch.tensor(voxel_box).to(points.device)
                    iou = boxes_iou3d_gpu(gt_box_i_tensor, voxel_box_tensor)
                    if iou >0:
                        feature = occ_feature[0, :, selected_voxels[j][2], selected_voxels[j][1], selected_voxels[j][0]]
                        label = {"label":int(gt_boxes[i, 7]), "frame_id":input_batch['frame_id'][b].item(), "z_idx":selected_voxels[j][2].item(), "y_idx":selected_voxels[j][1].item(), "x_idx":selected_voxels[j][0].item()}
                        feature_list.append(feature.cpu())
                        label_list.append(label)
                
        return feature_list, label_list
    
    # more than 10x faster than extract_features(). As it's difficult to vectorize the code over batches for this function, empirically bs=1 for each process works the best.
    def extract_features_faster_with_others(self, batch_dict, final_output_dir=None, distance_threshold=3.0, voxel_size=(0.05, 0.05, 0.1), min_bound=(0, -40, -3), max_bound=(70.4, 40, 1), downsample_ratios=(8.0, 8.0, 41.0/2)):
        
        min_bound = torch.tensor(min_bound, device=batch_dict['points'].device, dtype=batch_dict['points'].dtype)
        max_bound = torch.tensor(max_bound, device=batch_dict['points'].device, dtype=batch_dict['points'].dtype)
        voxel_size = torch.tensor(voxel_size, device=batch_dict['points'].device, dtype=batch_dict['points'].dtype)
        downsample_ratios = torch.tensor(downsample_ratios, device=batch_dict['points'].device, dtype=torch.float32)

        input_batch = batch_dict
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        occ_feature = self.backbone_3d.get_voxel_feature() # [B, 128, 2, 200, 176]

        feature_list = []
        label_list = []
        feature_list_others = []
        for b in range(occ_feature.shape[0]):
            points = input_batch['points'][input_batch['points'][:, 0]==b] # [num_points, 5], (batch_idx, x, y, z, intensity)
            points = points[:, 1:4] # [num_points, 3], (x, y, z)
            in_bounds_mask = (points >= min_bound).all(dim=1) & (points < max_bound).all(dim=1)
            points = points[in_bounds_mask]
            grid_indices = torch.floor((points - min_bound) / voxel_size / downsample_ratios).int()
            unique_grid_indices = torch.unique(grid_indices, dim=0) # [num_unique_voxels, 3], (x, y, z)
            unique_grid_indices_selected = torch.zeros((unique_grid_indices.shape[0]), dtype=torch.bool, device=points.device) # to record the selected voxels with labels, useful for latter randomly selecting voxels without labels
            voxel_centers = (unique_grid_indices + torch.tensor([0.5, 0.5, 0.5]).to(points.device)) * downsample_ratios * voxel_size + min_bound
            
            gt_boxes =  input_batch['gt_boxes'][b] 
            
            for i in range(gt_boxes.shape[0]):
                gt_box_i = gt_boxes[i, 0:7]
                gt_box_i_tensor = gt_box_i.unsqueeze(0).to(points.device) 
                distance = torch.norm(gt_box_i[0:3]-voxel_centers[:, 0:3], dim=1)
                selected_voxel_indices = torch.where(distance < distance_threshold)
                selected_voxel_centers = voxel_centers[selected_voxel_indices]
                selected_voxels = unique_grid_indices[selected_voxel_indices]
                for j, voxel_center in enumerate(selected_voxel_centers):
                    voxel_box = [[voxel_center[0], voxel_center[1], voxel_center[2], voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2], 0]]
                    voxel_box_tensor = torch.tensor(voxel_box).to(points.device)
                    iou = boxes_iou3d_gpu(gt_box_i_tensor, voxel_box_tensor)
                    if iou >0:
                        feature = occ_feature[b, :, selected_voxels[j][2], selected_voxels[j][1], selected_voxels[j][0]]
                        label = {"label":int(gt_boxes[i, 7]), "frame_id":input_batch['frame_id'][b].item(), "z_idx":selected_voxels[j][2].item(), "y_idx":selected_voxels[j][1].item(), "x_idx":selected_voxels[j][0].item()}
                        feature_list.append(feature.cpu())
                        label_list.append(label)
                        matches = (unique_grid_indices == selected_voxels[j]).all(dim=1)
                        position = matches.nonzero(as_tuple=True)[0]
                        unique_grid_indices_selected[position] = True
                
            # randomly select voxels without labels equals to the number of voxels with labels
            num_selected_voxels = unique_grid_indices_selected.sum().item()
            if num_selected_voxels > 0:
                voxels_wo_labels = unique_grid_indices[~unique_grid_indices_selected]
                selected_voxels_wo_labels = voxels_wo_labels[torch.randperm(voxels_wo_labels.shape[0])[:num_selected_voxels]]
                for voxel_wo_label in selected_voxels_wo_labels:
                    feature = occ_feature[b, :, voxel_wo_label[2], voxel_wo_label[1], voxel_wo_label[0]].unsqueeze(0)
                    feature_list_others.append(feature.cpu())
                   

        return feature_list, label_list, feature_list_others
        
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
