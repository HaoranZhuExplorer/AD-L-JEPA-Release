"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

# white, green, cyan, yellow
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def draw_occupancy(points, distance_threshold=10, iou_threshold=0, voxel_size=(0.05, 0.05, 0.1), grid_size=(176, 200, 40), min_bound=(0, -40, -3), max_bound=(70.4, 40, 1), gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, draw_origin=True):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    points = points[:, :3]
    
    # Calculate the actual bounds from the grid size and voxel size
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)

    # Ensure that points fall within the specified bounds
    in_bounds_mask = (points >= min_bound).all(axis=1) & (points <= max_bound).all(axis=1)
    points = points[in_bounds_mask]
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))

    downsample_ratios = [8.0, 8.0, 41.0/2]
    grid_indices = np.floor((points - min_bound) / voxel_size / downsample_ratios).astype(int)
    print("debug grid_indices", grid_indices[:, 0].min(), grid_indices[:, 0].max(), grid_indices[:, 1].min(), grid_indices[:, 1].max(), grid_indices[:, 2].min(), grid_indices[:, 2].max())

    # Calculate unique occupied voxels
    unique_grid_indices = np.unique(grid_indices, axis=0)
    print("debug unique_grid_indices", unique_grid_indices.shape)
    
    '''
    label_list = []
    # Visualize occupied voxels
    for voxel in unique_grid_indices:
        voxel_center = (voxel + [0.5, 0.5, 0.5]) * downsample_ratios * voxel_size + min_bound
        voxel_cube = open3d.geometry.TriangleMesh.create_box(width=voxel_size[0]*downsample_ratios[0], height=voxel_size[1]*downsample_ratios[1], depth=voxel_size[2]*downsample_ratios[2])
        # translate the voxel to the center by shifting half of the voxel size as open3d create a box in which the lower corner (not the geometric center!) is at the origin
        voxel_cube.translate(voxel_center - 0.5 * np.array([voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2]]))

        intersect = False
        intersect_label = None
        for i in range(gt_boxes.shape[0]):
            gt_box_i = np.array([gt_boxes[i, 0:7]])
            voxel_box = np.array([[voxel_center[0], voxel_center[1], voxel_center[2], voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2], 0]]) 
            gt_box_i_tensor = torch.from_numpy(gt_box_i).cuda().float()
            voxel_box_tensor = torch.from_numpy(voxel_box).cuda().float()
            distance = (gt_box_i_tensor[0][0:3]-voxel_box_tensor[0][0:3]).norm()
            if distance < distance_threshold:
                iou = boxes_iou3d_gpu(gt_box_i_tensor, voxel_box_tensor)
                if iou > iou_threshold:
                    print("debug iou", iou)
                    intersect = True
                    intersect_label = int(gt_boxes[i, 7])
                    break
        if intersect:
            label_list.append(intersect_label)
            voxel_cube.paint_uniform_color(box_colormap[intersect_label])
        else:
            voxel_cube.paint_uniform_color([1, 1, 1])  # White color for voxels
        vis.add_geometry(voxel_cube)

    print("debug intersect label_list", len(label_list), label_list)
    '''

    label_list = []
    voxel_centers = (unique_grid_indices + [0.5, 0.5, 0.5])* downsample_ratios * voxel_size + min_bound
    voxel_colors = np.zeros(unique_grid_indices.shape[0])
    # voxel_cube = open3d.geometry.TriangleMesh.create_box(width=voxel_size[0]*downsample_ratios[0], height=voxel_size[1]*downsample_ratios[1], depth=voxel_size[2]*downsample_ratios[2])
    # translate the voxel to the center by shifting half of the voxel size as open3d create a box in which the lower corner (not the geometric center!) is at the origin
    # voxel_cube.translate(voxel_center - 0.5 * np.array([voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2]]))
    
    for i in range(gt_boxes.shape[0]):
        gt_box_i = np.array([gt_boxes[i, 0:7]])
        gt_box_i_tensor = torch.from_numpy(gt_box_i).to("cuda").float()
        distance = torch.norm(gt_box_i_tensor[0, 0:3]-torch.tensor(voxel_centers[:, 0:3]).to("cuda"), dim=1)
        selected_voxel_indices = torch.where(distance < distance_threshold)[0].cpu().numpy()
        selected_voxel_centers = voxel_centers[selected_voxel_indices]
        selected_voxels = unique_grid_indices[selected_voxel_indices]
        for j, voxel_center in enumerate(selected_voxel_centers):
            voxel_box = [[voxel_center[0], voxel_center[1], voxel_center[2], voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2], 0]]
            voxel_box_tensor = torch.tensor(voxel_box).to("cuda").float()
            iou = boxes_iou3d_gpu(gt_box_i_tensor, voxel_box_tensor)
            if iou >0:
                label = int(gt_boxes[i, 7])
                label_list.append(label)
                voxel_colors[selected_voxel_indices[j]] = label
    
    for voxel_center in voxel_centers:
        voxel_cube = open3d.geometry.TriangleMesh.create_box(width=voxel_size[0]*downsample_ratios[0], height=voxel_size[1]*downsample_ratios[1], depth=voxel_size[2]*downsample_ratios[2])
        # translate the voxel to the center by shifting half of the voxel size as open3d create a box in which the lower corner (not the geometric center!) is at the origin
        voxel_cube.translate(voxel_center - 0.5 * np.array([voxel_size[0]*downsample_ratios[0], voxel_size[1]*downsample_ratios[1], voxel_size[2]*downsample_ratios[2]]))
        label = int(voxel_colors[np.where((voxel_centers == voxel_center).all(axis=1))[0]])
        if label != 0:
            voxel_cube.paint_uniform_color(box_colormap[label])
        else:
            voxel_cube.paint_uniform_color([1, 1, 1])
        vis.add_geometry(voxel_cube)
    print("debug intersect label_list", len(label_list), label_list)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
