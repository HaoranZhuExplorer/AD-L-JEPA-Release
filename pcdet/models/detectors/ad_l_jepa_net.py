from .detector3d_template_ad_l_jepa import Detector3DTemplate_AD_L_JEPA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
from tools.visual_utils import evaluate_features_utils

class AD_L_JEPA(Detector3DTemplate_AD_L_JEPA):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, plot=False, final_output_dir=None):
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
            loss, tb_dict, disp_dict = self.get_training_loss()

            if plot:
                print("debug frame_id", input_batch['frame_id'])
                print("debug loss", loss)
                print("debug tb_dict", tb_dict)
                
                gt_boxes = input_batch['gt_boxes'] # [B, M, 8]
                num_classes = len(torch.unique(gt_boxes[:,:,7]))

                # get recorded info
                context, prediction, target, coord_encoder, coord_target_encoder, encoder_indices, target_encoder_indices = self.backbone_3d.get_voxel_feature()
                bev_mask_encoder = self.backbone_3d.forward_re_dict['bev_mask_encoder'][0]
                bev_mask_target_encoder = self.backbone_3d.forward_re_dict['bev_mask_target_encoder'][0]
                bev_mask_encoder_empty = self.backbone_3d.forward_re_dict['bev_mask_encoder_empty'][0] 
                bev_mask_target_encoder_empty = self.backbone_3d.forward_re_dict['bev_mask_target_encoder_empty'][0]   
                empty_token = self.backbone_3d.empty_token
                bev_x_shape = context.shape[2]
                bev_y_shape = context.shape[3]

                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                bev_mask_encoder_map = np.full((bev_x_shape, bev_y_shape), np.nan)
                bev_mask_encoder_map[bev_mask_encoder.cpu().numpy()] = 1
                bev_mask_encoder_map[bev_mask_encoder_empty.cpu().numpy()] = 0
                plt.matshow(bev_mask_encoder_map)
                plt.axis('off') 
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) 
                plt.savefig(final_output_dir+"/bev_{}_mask_encoder.png".format(str(batch_dict['frame_id'])), bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.figure(figsize=(10, 10))
                bev_mask_target_encoder_map = np.full((bev_x_shape, bev_y_shape), np.nan)
                bev_mask_target_encoder_map[bev_mask_target_encoder.cpu().numpy()] = 1
                bev_mask_target_encoder_map[bev_mask_target_encoder_empty.cpu().numpy()] = 0
                plt.matshow(bev_mask_target_encoder_map)
                plt.axis('off') 
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) 
                plt.savefig(final_output_dir+"/bev_{}_mask_target_encoder.png".format(str(batch_dict['frame_id'])), bbox_inches='tight', pad_inches=0)
                plt.close()

                print("debug final_output_dir", final_output_dir)
                # viz input voxels for encoder and target encoder on feature map
                #evaluate_features_utils.get_selected_voxel_bev_map(coord_encoder, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_selected_voxels.png".format(str(batch_dict['frame_id'])))
                #evaluate_features_utils.get_selected_voxel_bev_map(coord_target_encoder, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_selected_target_voxels.png".format(str(batch_dict['frame_id'])))
                
                # viz binary clusters on feature map, evaluate pretrained features for geomatics
                if context is not None:
                    bev_feature_context = context.reshape(context.shape[0], 256, bev_x_shape, bev_y_shape) # [B, 256, 200, 176]
                    bev_feature_context = bev_feature_context.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                    bev_feature_context_origin = bev_feature_context.clone()
                    bev_feature_context = bev_feature_context.reshape(bev_feature_context.size(0), -1, bev_feature_context.size(3))  # shape: [batch_size, 200x176, 256] 
                    
                if target is not None:
                    bev_feature_target = target.reshape(target.shape[0], 256, bev_x_shape, bev_y_shape) # [B, 256, 200, 176]
                    bev_feature_target = bev_feature_target.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                    bev_feature_target_origin = bev_feature_target.clone()
                    bev_feature_target = bev_feature_target.reshape(bev_feature_target.size(0), -1, bev_feature_target.size(3))  # shape: [batch_size, 200x176, 256] 
                   
                if prediction is not None:    
                    bev_feature_prediction = prediction.reshape(prediction.shape[0], 256, bev_x_shape, bev_y_shape) # [B, 256, 200, 176]
                    bev_feature_prediction = bev_feature_prediction.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                    bev_feature_prediction_origin = bev_feature_prediction.clone()
                    bev_mask_target_encoder_all = torch.logical_or(bev_mask_target_encoder, bev_mask_target_encoder_empty)
                    bev_feature_prediction = bev_feature_prediction.squeeze(0)
                    bev_feature_prediction = bev_feature_prediction[bev_mask_target_encoder_all]
                    print("debug bev_feature_prediction", bev_feature_prediction.shape)
                    #bev_feature_prediction = bev_feature_prediction.reshape(bev_feature_prediction.size(0), -1, bev_feature_prediction.size(3))  # shape: [batch_size, 200x176, 256] 
                    cluster_result = evaluate_features_utils.viz_cluster_bev_feature(bev_feature_prediction, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_prediction.png".format(str(batch_dict['frame_id'])), 2) 
                    
                
                    cluster_map = np.full((bev_x_shape, bev_y_shape), np.nan)
                    cluster_map[bev_mask_target_encoder_all.cpu().numpy()] = cluster_result
                    iou_1 = np.logical_and(cluster_map == 1, bev_mask_target_encoder.cpu().numpy()).sum() / np.logical_or(cluster_map == 1, bev_mask_target_encoder.cpu().numpy()).sum()
                    iou_0 = np.logical_and(cluster_map == 0, bev_mask_target_encoder.cpu().numpy()).sum() / np.logical_or(cluster_map == 0, bev_mask_target_encoder.cpu().numpy()).sum()
                    iou_foreground = iou_1
                    if iou_0 > iou_1:
                        cluster_map_tmp = cluster_map.copy()
                        cluster_map_tmp[cluster_map == 1] = 0
                        cluster_map_tmp[cluster_map == 0] = 1
                        cluster_map = cluster_map_tmp
                        iou_foreground = iou_0
                    iou_empty = np.logical_and(cluster_map == 0, bev_mask_target_encoder_empty.cpu().numpy()).sum() / np.logical_or(cluster_map == 0, bev_mask_target_encoder_empty.cpu().numpy()).sum()

                    print("debug iou", iou_foreground, iou_empty)
                    plt.figure(figsize=(10, 10))
                    plt.matshow(cluster_map)
                    plt.axis('off') 
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) 
                    plt.savefig(final_output_dir+"/bev_{}_prediction_cluster_map.png".format(str(batch_dict['frame_id'])), bbox_inches='tight', pad_inches=0)
                    plt.close()




                # viz point clouds downsampled on feature map scale
                evaluate_features_utils.get_bev_map_kitti(input_batch['points'], final_output_dir+"/bev_{}_lidar_data.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_downsampled_lidar_data.png".format(str(batch_dict['frame_id'])))
                
                ## viz jepa loss on feature map scale
                #if prediction is not None and target is not None:
                #    loss_jepa_bkg_voxels, loss_jepa_foreground_voxels, loss_jepa_context_voxels, loss_jepa_target_voxels = evaluate_features_utils.viz_jepa_loss_bev(bev_feature_prediction_origin, bev_feature_target_origin, encoder_indices, target_encoder_indices, final_output_dir+"/bev_{}_jepa_loss.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_jepa_loss_bkg.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_jepa_loss_context.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_jepa_loss_target.png".format(str(batch_dict['frame_id'])))
                #    # viz histogram of jepa loss
                #    evaluate_features_utils.viz_histogram(loss_jepa_bkg_voxels, final_output_dir+"/bev_{}_jepa_loss_bkg_histogram.png".format(str(batch_dict['frame_id'])))
                #    evaluate_features_utils.viz_histogram(loss_jepa_context_voxels, final_output_dir+"/bev_{}_jepa_loss_context_histogram.png".format(str(batch_dict['frame_id'])))
                #    evaluate_features_utils.viz_histogram(loss_jepa_target_voxels, final_output_dir+"/bev_{}_jepa_loss_target_histogram.png".format(str(batch_dict['frame_id'])))

                # viz cosine similairty between prediction and target on feature map scale
                if prediction is not None and target is not None:
                    #cos_sim_bev_bkg, cos_sim_bev_foreground, cos_sim_bev_context, cos_sim_bev_target = evaluate_features_utils.viz_bev_cosine_similarity(bev_feature_prediction_origin, bev_feature_target_origin, bev_mask_target_encoder, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_cosine_similarity.png".format(str(batch_dict['frame_id'])), vmin=0.0, vmax=1.0)
                    cos_sim_bev_target, cos_sim_empty, cos_sim_bev_target_empty = evaluate_features_utils.viz_bev_cosine_similarity(bev_feature_prediction_origin, bev_feature_target_origin, bev_mask_target_encoder, bev_mask_target_encoder_empty, empty_token, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_cosine_similarity.png".format(str(batch_dict['frame_id'])), vmin=0.0, vmax=1.0)

                    cos_threshold = 0.5
                    cos_map = np.full((bev_x_shape, bev_y_shape), np.nan)
                    cos_map[(cos_sim_empty>cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()] = 0
                    cos_result = (cos_sim_empty<cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()
                    cos_map[(cos_sim_empty<cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()] = 1
                    iou_foreground = np.logical_and(cos_result, bev_mask_target_encoder.cpu().numpy()).sum() / np.logical_or(cos_result, bev_mask_target_encoder.cpu().numpy()).sum()
                    cos_result_empty = (cos_sim_empty>cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()
                    iou_empty = np.logical_and(cos_result_empty, bev_mask_target_encoder_empty.cpu().numpy()).sum() / np.logical_or(cos_result_empty, bev_mask_target_encoder_empty.cpu().numpy()).sum()
                    print("debug iou", iou_foreground, iou_empty)
                    plt.figure(figsize=(10, 10))
                    plt.matshow(cos_map)
                    plt.axis('off') 
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) 
                    plt.savefig(final_output_dir+"/bev_{}_prediction_cos_map.png".format(str(batch_dict['frame_id'])), bbox_inches='tight', pad_inches=0)
                    plt.close()


                    # viz histogram of cosine similarity
                    #evaluate_features_utils.viz_histogram(cos_sim_bev_bkg, final_output_dir+"/bev_{}_cosine_similarity_bkg_histogram.png".format(str(batch_dict['frame_id'])))
                    #evaluate_features_utils.viz_histogram(cos_sim_bev_context, final_output_dir+"/bev_{}_cosine_similarity_context_histogram.png".format(str(batch_dict['frame_id'])))
                    evaluate_features_utils.viz_histogram(cos_sim_bev_target, final_output_dir+"/bev_{}_cosine_similarity_target_histogram.png".format(str(batch_dict['frame_id'])))
                    evaluate_features_utils.viz_histogram(cos_sim_bev_target_empty, final_output_dir+"/bev_{}_cosine_similarity_target_empty_histogram.png".format(str(batch_dict['frame_id'])))
                    
                    print("debug cos_sim_bev_target", cos_sim_bev_target, np.mean(cos_sim_bev_target))
                    print("debug cos_sim_bev_target_empty", cos_sim_bev_target_empty, np.mean(cos_sim_bev_target_empty))

                #occ_k_means.bev_svd_analysis(bev_feature, final_output_dir+"/bev_{}_svd.png".format(str(batch_dict['frame_id'])))
            else:
                gt_boxes = input_batch['gt_boxes'] # [B, M, 8]
                num_classes = len(torch.unique(gt_boxes[:,:,7]))

                # get recorded info
                context, prediction, target, coord_encoder, coord_target_encoder, encoder_indices, target_encoder_indices = self.backbone_3d.get_voxel_feature()
                bev_mask_encoder = self.backbone_3d.forward_re_dict['bev_mask_encoder'][0]
                bev_mask_target_encoder = self.backbone_3d.forward_re_dict['bev_mask_target_encoder'][0]
                bev_mask_encoder_empty = self.backbone_3d.forward_re_dict['bev_mask_encoder_empty'][0] 
                bev_mask_target_encoder_empty = self.backbone_3d.forward_re_dict['bev_mask_target_encoder_empty'][0]   
                empty_token = self.backbone_3d.empty_token
                bev_x_shape = context.shape[2]
                bev_y_shape = context.shape[3]

                # viz binary clusters on feature map, evaluate pretrained features for geomatics
                if context is not None:
                    bev_feature_context = context.reshape(context.shape[0], 256, bev_x_shape, bev_y_shape) # [B, 256, 200, 176]
                    bev_feature_context = bev_feature_context.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                    bev_feature_context_origin = bev_feature_context.clone()
                    bev_feature_context = bev_feature_context.reshape(bev_feature_context.size(0), -1, bev_feature_context.size(3))  # shape: [batch_size, 200x176, 256] 
                    
                if target is not None:
                    bev_feature_target = target.reshape(target.shape[0], 256, bev_x_shape, bev_y_shape) # [B, 256, 200, 176]
                    bev_feature_target = bev_feature_target.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                    bev_feature_target_origin = bev_feature_target.clone()
                    bev_feature_target = bev_feature_target.reshape(bev_feature_target.size(0), -1, bev_feature_target.size(3))  # shape: [batch_size, 200x176, 256] 
                   
                if prediction is not None:    
                    bev_feature_prediction = prediction.reshape(prediction.shape[0], 256, bev_x_shape, bev_y_shape) # [B, 256, 200, 176]
                    bev_feature_prediction = bev_feature_prediction.permute(0, 2, 3, 1)  # shape: [batch_size, 200, 176, 256]
                    bev_feature_prediction_origin = bev_feature_prediction.clone()
                    bev_mask_target_encoder_all = torch.logical_or(bev_mask_target_encoder, bev_mask_target_encoder_empty)
                    bev_feature_prediction = bev_feature_prediction.squeeze(0)
                    bev_feature_prediction = bev_feature_prediction[bev_mask_target_encoder_all]
                    cluster_result = evaluate_features_utils.viz_cluster_bev_feature(bev_feature_prediction, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_prediction.png".format(str(batch_dict['frame_id'])), 2) 
                    
                    cluster_map = np.full((bev_x_shape, bev_y_shape), np.nan)
                    cluster_map[bev_mask_target_encoder_all.cpu().numpy()] = cluster_result
                    iou_1 = np.logical_and(cluster_map == 1, bev_mask_target_encoder.cpu().numpy()).sum() / np.logical_or(cluster_map == 1, bev_mask_target_encoder.cpu().numpy()).sum()
                    iou_0 = np.logical_and(cluster_map == 0, bev_mask_target_encoder.cpu().numpy()).sum() / np.logical_or(cluster_map == 0, bev_mask_target_encoder.cpu().numpy()).sum()
                    iou_foreground = iou_1
                    if iou_0 > iou_1:
                        cluster_map_tmp = cluster_map.copy()
                        cluster_map_tmp[cluster_map == 1] = 0
                        cluster_map_tmp[cluster_map == 0] = 1
                        cluster_map = cluster_map_tmp
                        iou_foreground = iou_0
                    iou_empty = np.logical_and(cluster_map == 0, bev_mask_target_encoder_empty.cpu().numpy()).sum() / np.logical_or(cluster_map == 0, bev_mask_target_encoder_empty.cpu().numpy()).sum()
                   
                    print("debug iou", iou_foreground, iou_empty)


                # viz point clouds downsampled on feature map scale
                #evaluate_features_utils.get_bev_map_waymo(input_batch['points'], final_output_dir+"/bev_{}_lidar_data.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_downsampled_lidar_data.png".format(str(batch_dict['frame_id'])))
                evaluate_features_utils.get_bev_map_kitti(input_batch['points'], final_output_dir+"/bev_{}_lidar_data.png".format(str(batch_dict['frame_id'])), final_output_dir+"/bev_{}_downsampled_lidar_data.png".format(str(batch_dict['frame_id'])))
                
                # viz cosine similairty between prediction and target on feature map scale
                # if prediction is not None and target is not None:
                #     #cos_sim_bev_bkg, cos_sim_bev_foreground, cos_sim_bev_context, cos_sim_bev_target = evaluate_features_utils.viz_bev_cosine_similarity(bev_feature_prediction_origin, bev_feature_target_origin, bev_mask_target_encoder, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_cosine_similarity.png".format(str(batch_dict['frame_id'])), vmin=0.0, vmax=1.0)
                #     cos_sim_bev_target, cos_sim_empty, cos_sim_bev_target_empty = evaluate_features_utils.viz_bev_cosine_similarity(bev_feature_prediction_origin, bev_feature_target_origin, bev_mask_target_encoder, bev_mask_target_encoder_empty, empty_token, bev_x_shape, bev_y_shape, final_output_dir+"/bev_{}_cosine_similarity.png".format(str(batch_dict['frame_id'])), vmin=0.0, vmax=1.0)
                    
                #     cos_threshold = 0.5
                #     cos_map = np.full((bev_x_shape, bev_y_shape), np.nan)
                #     cos_map[(cos_sim_empty>cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()] = 0
                #     cos_result = (cos_sim_empty<cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()
                #     cos_map[(cos_sim_empty<cos_threshold) & bev_mask_target_encoder_all.cpu().numpy()] = 1
                #     iou_foreground = np.logical_and(cos_result, bev_mask_target_encoder.cpu().numpy()).sum() / np.logical_or(cos_result, bev_mask_target_encoder.cpu().numpy()).sum()
                #     iou_empty = np.logical_and(cos_result, bev_mask_target_encoder_empty.cpu().numpy()).sum() / np.logical_or(cos_result, bev_mask_target_encoder_empty.cpu().numpy()).sum()
                #     print("debug iou", iou_foreground, iou_empty)

            return loss
        

    def get_training_loss(self):
        disp_dict = {} 

        loss, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_pretrain': loss.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict