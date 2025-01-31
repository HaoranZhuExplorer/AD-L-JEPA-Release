from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import torch
import torch.nn.functional as F

def get_bev_map_kitti(points, output_path, downsampled_output_path):
    points = points[:, 1:].cpu().numpy()  # Remove batch_idx, convert from tensor to numpy array

    # Define the bounds and resolution of the BEV image
    x_range = (0, 70.4)
    y_range = (-40, 40)
    z_range = (-3, 1)
    resolution = 0.05

    # Filter points within the bounds
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
           (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    points = points[mask]

    # Convert points to BEV coordinates
    x_img = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    y_img = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)
    x_img = np.clip(x_img, 0, 1408 - 1)
    y_img = np.clip(y_img, 0, 1600 - 1)

    # Normalize intensity values
    intensity = points[:, 3]  # Assuming the fourth column is intensity
    intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    # Create BEV map, initialized to white
    lidar_bev_data = np.full((1600, 1408, 3), 255, dtype=np.uint8)  # Initialize as white

    # Map intensity to colors
    cmap = cm.get_cmap('viridis')  # Choose any colormap
    colors = cmap(intensity_normalized)[:, :3]  # Get RGB values

    for i in range(len(x_img)):
        lidar_bev_data[y_img[i], x_img[i]] = (colors[i] * 255).astype(np.uint8)  # Scale colors to [0, 255]

    #plt.figure(figsize=(10, 10), dpi=100)
    plt.figure()
    plt.imshow(lidar_bev_data)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    lidar_bev_data_origin = np.ones((1600, 1408, 3), dtype=np.uint8) * 255  # Initialize as white
    lidar_bev_data_origin[y_img, x_img] = 0  # Points are black

    # Calculate the downsampling factors
    downsample_size = (200, 176)
    factor_x = int(lidar_bev_data_origin.shape[0] / downsample_size[0])
    factor_y = int(lidar_bev_data_origin.shape[1] / downsample_size[1])

    # Downsample the BEV map
    downsampled_lidar_bev_data = lidar_bev_data_origin.reshape(downsample_size[0], factor_x, downsample_size[1], factor_y, 3).min(axis=(1, 3))
    #plt.figure(figsize=(10, 10), dpi=100)
    plt.figure()
    plt.imshow(downsampled_lidar_bev_data)
    plt.axis('off')
    plt.savefig(downsampled_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_bev_map_waymo(points, output_path, downsampled_output_path):
    points = points[:, 1:].cpu().numpy()  # Remove batch_idx, convert from tensor to numpy array

    # Define the bounds and resolution of the BEV image
    x_range = (-75.2, 75.2)
    y_range = (-75.2, 75.2)
    z_range = (-2, 4)
    resolution = 0.1

    # Filter points within the bounds
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
           (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    points = points[mask]

    # Convert points to BEV coordinates
    x_img = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    y_img = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)
    x_img = np.clip(x_img, 0, 1504 - 1)
    y_img = np.clip(y_img, 0, 1504 - 1)

    # Normalize intensity values
    intensity = points[:, 3]  # Assuming the fourth column is intensity
    intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    # Create BEV map, initialized to white
    lidar_bev_data = np.full((1504, 1504, 3), 255, dtype=np.uint8)  # Initialize as white

    # Map intensity to colors
    cmap = cm.get_cmap('viridis')  # Choose any colormap
    colors = cmap(intensity_normalized)[:, :3]  # Get RGB values

    for i in range(len(x_img)):
        lidar_bev_data[y_img[i], x_img[i]] = (colors[i] * 255).astype(np.uint8)  # Scale colors to [0, 255]

    plt.figure(figsize=(10, 10))
    plt.imshow(lidar_bev_data, aspect='auto')
    plt.axis('off')  
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # Calculate the downsampling factors
    downsample_size = (188, 188)
    factor_x = int(lidar_bev_data.shape[0] / downsample_size[0])
    factor_y = int(lidar_bev_data.shape[1] / downsample_size[1])
    # Downsample the BEV map
    downsampled_lidar_bev_data = lidar_bev_data.reshape(downsample_size[0], factor_x, downsample_size[1], factor_y, 3).max(axis=(1, 3))
    plt.figure(figsize=(10, 10))
    plt.imshow(downsampled_lidar_bev_data)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(downsampled_output_path)
    plt.close()

def get_selected_voxel_bev_map(coord, bev_x_shape, bev_y_shape, output_path):
    coord = coord[:, 1:].cpu().numpy().astype(int) # remove batch_idx, convert from tensor to numpy array 

    # Create BEV map
    voxel_bev_data = np.zeros((bev_x_shape*8, bev_y_shape*8, 3), dtype=np.uint8)
    voxel_bev_data[coord[:,1], coord[:,2]] = 255  # Points are white

    #plt.figure(figsize=(10, 10))
    #plt.imshow(voxel_bev_data)
    #plt.title('selected voxels')
    #plt.savefig(output_path)
    #plt.close()

    # Calculate the downsampling factors
    downsample_size = (bev_x_shape, bev_y_shape)
    factor_x = int(voxel_bev_data.shape[0] / downsample_size[0])
    factor_y = int(voxel_bev_data.shape[1] / downsample_size[1])
    # Downsample the BEV map
    downsampled_voxel_bev_data = voxel_bev_data.reshape(downsample_size[0], factor_x, downsample_size[1], factor_y, 3).max(axis=(1, 3))
    plt.figure(figsize=(10, 10))
    plt.imshow(downsampled_voxel_bev_data)
    plt.title('downsampled selected voxels')
    plt.savefig(output_path)
    plt.close()


# to do: analysis for bev feature, e.g., svd
def bev_svd_analysis(features, output_path):
    features = features.squeeze()
    features_norm = torch.norm(features, dim=1)
    print("non zero grids number", torch.sum(features_norm>0))
    
def viz_cluster_bev_feature(features, bev_x_shape, bev_y_shape, output_path, n_clusters=2):
    """Visualize bev feature"""

    # Assuming the input tensor is given as `features` with shape [1, 200x176, 256]
    # Generate a random tensor for demonstration; replace this with your actual tensor.
    # Reshape the feature tensor from [1, 200x176, 256] to [200x176, 256]
    features = features.cpu()
    features_reshaped = features.view(-1, 256).numpy()

    # Initialize KMeans clustering
    # Here `n_clusters` can be chosen based on the expected number of distinct objects or through analysis
    n_clusters = n_clusters  # Example: change this number based on your specific needs
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)

    # Fit the KMeans algorithm on the flattened feature array
    kmeans.fit(features_reshaped)

    # Output the cluster labels
    labels = kmeans.labels_
    print("debug labels", labels.shape)

    return labels

    # # Reshape labels to match the 2D grid (200x176)
    # labels_2d = labels.reshape(bev_x_shape, bev_y_shape)

    # # Plotting the cluster labels on a 2D grid
    # plt.figure(figsize=(10, 10))
    # plt.imshow(labels_2d, cmap='viridis', interpolation='nearest')
    # #plt.colorbar(label='Cluster Label')
    # plt.title('Clustering of 256-D BEV Features on {}x{} Grid'.format(bev_x_shape, bev_y_shape))
    # plt.savefig(output_path)
    # plt.close()
    
def viz_jepa_loss_bev(prediction, target, encoder_indices, target_encoder_indices, output_path, output_path_bkg, output_path_context, output_path_target):
    downsample_factor = 8 
    #target = F.normalize(target, p=2, dim=3)
    #prediction = F.normalize(prediction, p=2, dim=3)
    loss_jepa_bev = torch.mean(torch.abs(prediction-target), dim=3)
    loss_jepa_bev = loss_jepa_bev.squeeze().cpu().numpy()
    
    encoder_indices = encoder_indices[:, 1:].cpu().numpy() # remove batch_idx, convert from tensor to numpy array 
    target_encoder_indices = target_encoder_indices[:, 1:].cpu().numpy() # remove batch_idx, convert from tensor to numpy array 

     # Calculate the loss for the empty voxels
    indices = torch.zeros(loss_jepa_bev.shape)
    indices[encoder_indices[:, 1], encoder_indices[:, 2]] = 1
    indices[target_encoder_indices[:, 1], target_encoder_indices[:, 2]] = 2
    loss_jepa_bkg_voxels = loss_jepa_bev[indices==0]
    loss_jepa_foreground_voxels = loss_jepa_bev[indices!=0]
    loss_jepa_context_voxels = loss_jepa_bev[indices==1]
    loss_jepa_target_voxels = loss_jepa_bev[indices==2]

    print("debug empty, occupied encoder voxels, occupied target encoder voxels", torch.sum(indices==0), torch.sum(indices==1), torch.sum(indices==2))
    print("debug loss", "loss jepa avg", np.mean(loss_jepa_bev), "bkg avg", np.mean(loss_jepa_bkg_voxels), "bkg max", np.max(loss_jepa_bkg_voxels), "bkg median", np.median(loss_jepa_bkg_voxels), "context mean", np.mean(loss_jepa_context_voxels), "context max", np.max(loss_jepa_context_voxels), "context median", np.median(loss_jepa_context_voxels), "target mean", np.mean(loss_jepa_target_voxels), "target max", np.max(loss_jepa_target_voxels), "target median", np.median(loss_jepa_target_voxels))
    plt.figure(figsize=(10, 10))
    #plt.matshow(loss_jepa_bev,vmin = 0, vmax = 0.005)
    plt.matshow(loss_jepa_bev)
    plt.colorbar()
    plt.title('jepa loss')
    plt.savefig(output_path)
    plt.close()

    loss_jepa_bev_bkg = loss_jepa_bev.copy()
    loss_jepa_bev_bkg[indices!=0] = np.nan
    plt.figure(figsize=(10, 10))
    #plt.matshow(loss_jepa_bev_bkg, vmin = 0, vmax = 0.005)
    plt.matshow(loss_jepa_bev_bkg)
    plt.colorbar()
    plt.title('jepa loss')
    plt.savefig(output_path_bkg)
    plt.close()

    loss_jepa_bev_context = loss_jepa_bev.copy()
    loss_jepa_bev_context[indices!=1] = np.nan
    plt.figure(figsize=(10, 10))
    #plt.matshow(loss_jepa_bev_context, vmin = 0, vmax = 0.005)
    plt.matshow(loss_jepa_bev_context)
    plt.colorbar()
    plt.title('jepa loss')
    plt.savefig(output_path_context)
    plt.close()

    loss_jepa_bev_target = loss_jepa_bev.copy()
    loss_jepa_bev_target[indices!=2] = np.nan
    plt.figure(figsize=(10, 10))
    #plt.matshow(loss_jepa_bev_target, vmin = 0, vmax = 0.005)
    plt.matshow(loss_jepa_bev_target)
    plt.colorbar()
    plt.title('jepa loss')
    plt.savefig(output_path_target)
    plt.close()

    return loss_jepa_bkg_voxels, loss_jepa_foreground_voxels, loss_jepa_context_voxels, loss_jepa_target_voxels




def viz_bev_cosine_similarity(prediction, target, bev_mask_target_encoder, bev_mask_target_encoder_empty, empty_token, bev_x_shape, bev_y_shape, output_path,  vmin=0.0, vmax=1.0):
    target = target.squeeze()
    prediction = prediction.squeeze()

    # cos_sim now holds cosine similarity with special handling for zero vectors
    cos_sim_bev = F.cosine_similarity(target, prediction, dim=2)
    print("debug cos sim", cos_sim_bev.shape, bev_mask_target_encoder.shape)

    cos_sim_bev = cos_sim_bev.cpu().numpy()
    cos_sim_bev[bev_mask_target_encoder.cpu().numpy()==0] = np.nan

    plt.figure(figsize=(10, 10))
    plt.matshow(cos_sim_bev,  vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('cosine similarity')
    plt.savefig(output_path)
    plt.close()

    print("debug prediction", prediction.shape, bev_mask_target_encoder_empty.shape, empty_token.shape)

    empty_token = empty_token.squeeze()
    
    cos_sim_empty = F.cosine_similarity(prediction.view(-1, 256), empty_token, dim=-1).view(bev_x_shape, bev_y_shape)
    cos_sim_empty = cos_sim_empty.cpu().numpy()
    bev_mask_target_encoder_all = torch.logical_or(bev_mask_target_encoder, bev_mask_target_encoder_empty)
    cos_sim_empty[bev_mask_target_encoder_all.cpu().numpy()==0] = np.nan
    
    #plt.figure(figsize=(10, 10))
    plt.figure()
    plt.matshow(cos_sim_empty,  vmin=vmin, vmax=vmax)
    plt.axis('off')
    cbar = plt.colorbar()
    # Set the colorbar label font size
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    #plt.savefig(output_path.replace('.png', '_empty.pdf'))
    
    #plt.title('cosine similarity')
    
    plt.savefig(output_path.replace('.png', '_empty.pdf'), bbox_inches='tight', pad_inches=1)
    plt.close()
    
    return cos_sim_bev[bev_mask_target_encoder.cpu().numpy()], cos_sim_empty, cos_sim_empty[bev_mask_target_encoder_empty.cpu().numpy()]
    #return cos_sim_bev[bev_mask_target_encoder.cpu().numpy()], cos_sim_empty.flatten()

   

def viz_cosine_similarity_vs_loss(cos_sim_bev, loss_jepa_bev, output_path):
    plt.figure(figsize=(10, 10))
    plt.scatter(cos_sim_bev, loss_jepa_bev)
    plt.xlabel('cosine similarity')
    plt.ylabel('jepa loss')
    plt.title('cosine similarity vs jepa loss')
    plt.savefig(output_path)
    plt.close()

def viz_histogram(data, output_path):
    plt.figure(figsize=(10, 10))
    plt.hist(data, bins=100)
    plt.title('histogram')
    plt.savefig(output_path)
    plt.close()

def viz_bev_l2_norm(input, encoder_indices, target_encoder_indices, output_path, output_path_bkg, output_path_context, output_path_target,  output_path_bkg_hist, output_path_context_hist, output_path_target_hist, vmin=0.0, vmax=0.005):
    input = input.squeeze()
    #input = F.normalize(input, p=2, dim=2)
    norm = torch.norm(input, dim=2).cpu().numpy()

    mean = np.mean(norm)
    std = np.std(norm)
    vmin = mean - 2 * std
    vmax = mean + 2 * std
    #print("debug viz l2 norm avg", mean, std)
    
    plt.figure(figsize=(10, 10))
    plt.matshow(norm)
    #plt.matshow(norm,  vmin=0, vmax=mean)
    plt.colorbar()
    plt.title('l2 norm')
    plt.savefig(output_path)
    plt.close()

    encoder_indices = encoder_indices[:, 1:].cpu().numpy() # remove batch_idx, convert from tensor to numpy array 
    target_encoder_indices = target_encoder_indices[:, 1:].cpu().numpy() # remove batch_idx, convert from tensor to numpy array 
    indices = torch.zeros(norm.shape)
    indices[encoder_indices[:, 1], encoder_indices[:, 2]] = 1
    indices[target_encoder_indices[:, 1], target_encoder_indices[:, 2]] = 2

    norm_bkg = norm.copy()
    norm_bkg[indices!=0] = np.nan
    mean = np.mean(norm_bkg[indices==0])
    std = np.std(norm_bkg[indices==0])
    vmin = mean - 2 * std
    vmax = mean + 2 * std
    print("debug viz l2 norm bkg", mean, std)
    plt.figure(figsize=(10, 10))
    #plt.matshow(norm_bkg,  vmin=0, vmax=mean)
    plt.matshow(norm_bkg)
    plt.colorbar()
    plt.title('l2 norm bkg')
    plt.savefig(output_path_bkg)
    plt.close()

    norm_context = norm.copy()
    norm_context[indices!=1] = np.nan
    mean = np.mean(norm_context[indices==1])
    std = np.std(norm_context[indices==1])
    vmin = mean - 2 * std
    vmax = mean + 2 * std
    print("debug viz l2 norm context", mean, std)
    plt.figure(figsize=(10, 10))
    #plt.matshow(norm_context,  vmin=0, vmax=mean)
    plt.matshow(norm_context)
    plt.colorbar()
    plt.title('l2 norm context')
    plt.savefig(output_path_context)
    plt.close()

    norm_target = norm.copy()
    norm_target[indices!=2] = np.nan
    mean = np.mean(norm_target[indices==2])
    std = np.std(norm_target[indices==2])
    vmin = mean - 2 * std
    vmax = mean + 2 * std
    #print("debug viz l2 norm target", mean, std)
    #plt.figure(figsize=(10, 10))
    #plt.matshow(norm_target,  vmin=0, vmax=mean)
    plt.matshow(norm_target)
    plt.colorbar()
    plt.title('l2 norm target')
    plt.savefig(output_path_target)
    plt.close()

    print("debug l2 norm output path", output_path)
    print("debug l2 norm", "norm jepa avg", np.mean(norm), "bkg avg", np.mean(norm[indices==0]), "bkg max", np.max(norm[indices==0]), "bkg median", np.median(norm[indices==0]), "context mean", np.mean(norm[indices==1]), "context max", np.max(norm[indices==1]), "context median", np.median(norm[indices==1]), "target mean", np.mean(norm[indices==2]), "target max", np.max(norm[indices==2]), "target median", np.median(norm[indices==2]))


    # viz histogram of l2 norm of context, prediction and target
    plt.figure(figsize=(10, 10))
    plt.hist(norm[indices==0], bins=100)
    plt.title('histogram')
    plt.savefig(output_path_bkg_hist)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.hist(norm[indices==1], bins=100)
    plt.title('histogram')
    plt.savefig(output_path_context_hist)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.hist(norm[indices==2], bins=100)
    plt.title('histogram')
    plt.savefig(output_path_target_hist)
    plt.close()