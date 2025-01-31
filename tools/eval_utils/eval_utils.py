import pickle
import time

import numpy as np
import torch
import torch.nn.functional
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import torch.distributed as dist
import os
import json

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    
    #det_annos = pickle.load(open(result_dir / 'result.pkl', 'rb'))

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    with open(result_dir / 'metrics.pkl', 'wb') as f2:
        pickle.dump(result_dict, f2)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    main_metric = {}
    main_metric['Car_3d/moderate']=ret_dict['Car_3d/moderate']
    main_metric['Car_3d/moderate_R40'] = ret_dict['Car_3d/moderate_R40']
    main_metric['Pedestrian_3d/moderate'] = ret_dict['Pedestrian_3d/moderate']
    main_metric['Pedestrian_3d/moderate_R40'] = ret_dict['Pedestrian_3d/moderate_R40']
    main_metric['Cyclist_3d/moderate'] = ret_dict['Cyclist_3d/moderate']
    main_metric['Cyclist_3d/moderate_R40'] = ret_dict['Cyclist_3d/moderate_R40']
    main_metric['mAP_3d/moderate'] = (ret_dict['Car_3d/moderate'] + ret_dict['Pedestrian_3d/moderate'] + ret_dict['Cyclist_3d/moderate']) / 3
    main_metric['mAP_3d/moderate_R40'] = (ret_dict['Car_3d/moderate_R40'] + ret_dict['Pedestrian_3d/moderate_R40'] + ret_dict['Cyclist_3d/moderate_R40']) / 3
    json.dump(main_metric, open(result_dir / 'main_metric.json', 'w'))

    return ret_dict


def eval_one_epoch_modified(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    rpn_loss = 0.0
    rpn_loss_cls = 0.0
    rpn_loss_loc = 0.0
    rpn_loss_dir = 0.0

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            if dist_test:
                loss, loss_dict, _ = model.module.get_training_loss()
            else:
                loss, loss_dict, _ = model.get_training_loss()
            rpn_loss += loss_dict['rpn_loss']
            rpn_loss_cls += loss_dict['rpn_loss_cls']
            rpn_loss_loc += loss_dict['rpn_loss_loc']
            rpn_loss_dir += loss_dict['rpn_loss_dir']
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    rpn_loss /= len(dataloader)
    rpn_loss_cls /= len(dataloader)
    rpn_loss_loc /= len(dataloader)
    rpn_loss_dir /= len(dataloader)
    result_loss_dict = {"rpn_loss":rpn_loss, "rpn_loss_cls":rpn_loss_cls, "rpn_loss_loc":rpn_loss_loc, "rpn_loss_dir":rpn_loss_dir}
    
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    with open(result_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    with open(result_dir / 'loss.pkl', 'wb') as f:
        pickle.dump(result_loss_dict, f)

    print("log from eval_utils: result_loss_dict", result_loss_dict)
    print("log from eval_utils: metrics", result_dict)
    

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    
    return ret_dict


def eval_ssl_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)


    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model(batch_dict, True, str(final_output_dir))
            #model(batch_dict, False, str(final_output_dir))
            print("debug log from eval_utils: iter", i)
            if i == 50:
                break


def extract_knn_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, training=True):
    def save_temp_results(features, labels, rank, output_dir):
        temp_file_features = output_dir / f"temp_features_{rank}.pt"
        torch.save(features, temp_file_features)
        temp_file_labels = output_dir / f"temp_labels_{rank}.pt"
        torch.save(labels, temp_file_labels)

    def gather_results(num_processes, result_dir, training=True):
        all_results_features = []
        for rank in range(num_processes):
            temp_file = result_dir / f"temp_features_{rank}.pt"
            results = torch.load(temp_file)
            all_results_features.extend(results)
            os.remove(temp_file)
        all_results_features = torch.stack(all_results_features)
        if training:
            final_results_file = result_dir / "train_features.pt"
        elif not training:
            final_results_file = result_dir / "val_features.pt"
        torch.save(all_results_features, final_results_file)

        all_results_labels = []
        for rank in range(num_processes):
            temp_file = result_dir / f"temp_labels_{rank}.pt"
            results = torch.load(temp_file)
            all_results_labels.extend(results)
            os.remove(temp_file)
        if training:
            final_results_file = result_dir / "train_labels.pt"
        elif not training:
            final_results_file = result_dir / "val_labels.pt"
        torch.save(all_results_labels, final_results_file)

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)


    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    feature_list_all = []
    label_list_all = []
    for i, batch_dict in enumerate(tqdm.tqdm(dataloader)):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            if dist_test:
                feature_list, label_list = model.module.extract_features_faster(batch_dict, str(final_output_dir), distance_threshold=3.0, voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'], min_bound=cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0:3], max_bound=cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:6], downsample_ratios=(8.0, 8.0, 41.0/2))
            else:
                feature_list, label_list = model.extract_features_faster(batch_dict, str(final_output_dir), distance_threshold=3.0, voxel_size=cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'], min_bound=cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0:3], max_bound=cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:6], downsample_ratios=(8.0, 8.0, 41.0/2))
 
            feature_list_all.extend(feature_list)
            label_list_all.extend(label_list)
    
    if dist_test:
        save_temp_results(feature_list_all, label_list_all, local_rank, result_dir)
        dist.barrier()
        if local_rank == 0:
            gather_results(num_gpus, result_dir, training=training)
    else:
        if training:
            final_feature_file = result_dir / "train_features.pt"
            torch.save(feature_list_all, final_feature_file)
            final_label_file = result_dir / "train_labels.pt"
            torch.save(label_list_all, final_label_file)
        elif not training:
            final_feature_file = result_dir / "val_features.pt"
            torch.save(feature_list_all, final_feature_file)
            final_label_file = result_dir / "val_labels.pt"
            torch.save(label_list_all, final_label_file)


def select_k_features_per_class(features, labels, K):
    """
    Randomly select K features per class from the feature matrix and labels.

    Parameters:
    - features: torch.Tensor of shape (128, N), where 128 is the feature dimension and N is the number of samples.
    - labels: torch.Tensor of shape (N,), containing class labels corresponding to the features.
    - K: int, the number of features to select per class.

    Returns:
    - selected_features: torch.Tensor of shape (128, K * M), where M is the number of unique classes.
    - selected_labels: torch.Tensor of shape (K * M,), the labels corresponding to the selected features.
    """
    unique_classes = torch.unique(labels)
    selected_features = []
    selected_labels = []

    for cls in unique_classes:
        # Get the indices of features belonging to the current class
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        
        # Check if there are enough samples in the class
        if len(class_indices) < K:
            raise ValueError(f"Not enough samples in class {cls.item()} to select {K} features.")
        
        # Randomly sample K indices
        sampled_indices = torch.randperm(len(class_indices))[:K]
        sampled_indices = class_indices[sampled_indices]
        
        # Collect the selected features and labels
        selected_features.append(features[:, sampled_indices])
        selected_labels.append(labels[sampled_indices])

    # Concatenate results along the second dimension for features and flatten labels
    selected_features = torch.cat(selected_features, dim=1)
    selected_labels = torch.cat(selected_labels)

    return selected_features, selected_labels

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T=0.07, num_classes=3):
    # train_features: [N, 128]
    # train_labels: [{"label":int, "frame_id":int, "z_idx":int, "y_idx":int, "x_idx":int}, {}, ...] of length N
    # test_features: [M, 128]
    # test_labels: [{"label":int, "frame_id":int, "z_idx":int, "y_idx":int, "x_idx":int}, {}, ...] of length M
    # k: number of neighbors
    # T: temperature
    # num_classes: number of classes. For KITTI, num_classes=3

    top1_class = torch.zeros(num_classes).to(train_features.device)  # Top-1 accuracy per class
    total_class = torch.zeros(num_classes).to(train_features.device)  # Total per class
    train_features = train_features.t()
    train_labels = torch.tensor([target["label"] for target in train_labels]).to(train_features.device) # [N]
    train_labels = train_labels-1 # 0-indexed
    print("debug", train_features.shape, train_labels.shape, torch.sum(train_labels==0), torch.sum(train_labels==1), torch.sum(train_labels==2))
    train_features, train_labels = select_k_features_per_class(train_features, train_labels, 10000)
    print("debug", train_features.shape, train_labels.shape, torch.sum(train_labels==0), torch.sum(train_labels==1), torch.sum(train_labels==2))
   
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    # Function to compute singular values and plot the spectrum
    def plot_singular_values(tensor):
        # Compute the singular values using torch.svd
        # Plot the cumulative explained variance
        c_svd = np.cumsum(S.numpy()) / S.numpy().sum()
        plt.figure(figsize=(3 * 0.95, 3* 0.95), dpi=250)
        plt.plot(c_svd, label=f"AD-JEPA")
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Cumulative Explained Variance')
        plt.title('AD-JEPA')
        plt.legend()
        plt.show()

    def auc(singular_values):
        # Equation 2 from https://arxiv.org/abs/2209.15007
        explvar = np.cumsum(singular_values) / singular_values.sum()
        return explvar.sum() / len(explvar)
    

    cov_loss = off_diagonal(torch.cov(train_features.T)).pow_(2).sum().div(
            train_features.shape[0]
        )
    print("cov loss ad_l_jepa", cov_loss)
    U, S, V = torch.svd(torch.cov(train_features.T))

    # Plot the singular values (spectrum)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(S.numpy()/S.numpy()[0], marker='o', label='')
    plt.title('Singular Value Spectrum')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.yscale('log')
    plt.yticks([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("svd.png")
    plt.close()

    # Plot the cumulative explained variance
    c_svd_ad_l_jepa = np.cumsum(S.numpy()) / S.numpy().sum()
    plt.figure(figsize=(10, 6))
    plt.plot(c_svd_ad_l_jepa, marker='o', label='')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.savefig("csvd.png")
    plt.close()

    # Compute the AUC
    auc_ad_l_jepa = auc(S.numpy())
    print(f'AUC AD-L-JEPA: {auc_ad_l_jepa}')
    exit()


    # # TSNE visualization
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # class_names = [
    #     "Vehicle", "Pedestrian", "Cyclist"
    # ]

    # # Ensure labels are numpy array for processing
    # labels_np = train_labels.cpu().numpy()
    # unique_labels = np.unique(labels_np)
    # unique_class_names = [class_names[label-1] for label in unique_labels]  # Adjust class names based on unique_labels
    # label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    # mapped_labels = np.array([label_mapping[label] for label in labels_np])

    # data_np = train_features.t().cpu().numpy()
    # mapped_labels_np = mapped_labels

    # tsne = TSNE(n_components=2, perplexity=15, learning_rate=10, verbose=1)
    # data_2d = tsne.fit_transform(data_np)

    # plt.figure(figsize=(8, 8))
    # scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=mapped_labels_np, cmap='tab20', s=1, alpha=0.6)
    # plt.title('t-SNE visualization of BEVContrast embeddings')
    # plt.xlabel('t-SNE dim 1')
    # plt.ylabel('t-SNE dim 2')
    # plt.savefig("tsne.png")
    # plt.close()



    num_test_images, num_chunks = len(test_labels), 25000 # to avoid OOM; 2500 works well on 12GB 1080Ti
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in tqdm.tqdm(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        targets = torch.tensor([target["label"] for target in targets]).to(train_features.device) # [M]
        targets = targets-1 # 0-indexed
        batch_size = targets.shape[0]
        # l2 normalization
        features = torch.nn.functional.normalize(features, dim=1)
        train_features = torch.nn.functional.normalize(train_features, dim=0)
        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features) # (batch_size, N)
        distances, indices = similarity.topk(k, largest=True, sorted=True) # distances: (batch_size, k); indices: (batch_size, k) 
        candidates = train_labels.view(1, -1).expand(batch_size, -1) # (batch_size, N)
        retrieved_neighbors = torch.gather(candidates, 1, indices) # (batch_size, k)
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_() # (batch_size * k, num_classes)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1) # (batch_size * k, num_classes)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.max(1)

        # Update accuracy per class
        for class_idx in range(num_classes):
            correct = predictions[targets == class_idx] == targets[targets == class_idx]
            top1_class[class_idx] += correct.float().sum().item()
            total_class[class_idx] += (targets == class_idx).sum().item()

        # Calculate per-class and overall accuracy
        top1_class_accuracy = (top1_class / total_class) * 100
        average_accuracy = top1_class_accuracy.mean().item()

        print("debug", top1_class_accuracy, average_accuracy)


    return top1_class_accuracy.tolist(), average_accuracy, top1_class.tolist(), total_class.tolist()
    


if __name__ == '__main__':
    pass