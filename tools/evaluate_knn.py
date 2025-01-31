# usage:
# bash ./scripts/dist_evaluate_knn ${NUM_GPUS} --cfg_file cfgs/kitti_models/second.yaml --batch_size ${BATCH_SIZE} --pretrained_model ${PRETRAINED_MODEL} --extra_tag $EXTRA_TAG$
# e.g.:
# bash ./scripts/dist_evaluate_knn.sh 4 --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --pretrained_model ../output/kitti_models/ad_l_jepa_kitti/ad_l_jepa_exp_14_2_run_2/ckpt/checkpoint_epoch_30.pth --extra_tag knn_ad_l_jepa_exp_14_2_run_2
# bash ./scripts/dist_evaluate_knn.sh 4 --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --pretrained_model ../output/kitti_models/voxel_mae_kitti/default/ckpt/checkpoint_epoch_30.pth --extra_tag knn_occupancy_mae
# bash ./scripts/dist_evaluate_knn.sh 4 --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --ckpt ../output/kitti_models/second/scratch_run_1_saving_all_epochs/ckpt/checkpoint_epoch_80.pth --extra_tag knn_second_scratch_run_1
# bash ./scripts/dist_evaluate_knn.sh 4 --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --ckpt ../output/kitti_models/second/second_ad_l_jepa_exp_14_2_new_epoch_30_run_1/ckpt/checkpoint_epoch_80.pth --extra_tag knn_second_ad_l_jepa_exp_14_2_new_epoch_30_run_1_test

import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import random
import json

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, data_loader, args, eval_output_dir, logger, epoch_id, dist_test=False, training=True):
    # load checkpoint
    if args.pretrained_model is not None and 'ad_l_jepa' in args.pretrained_model and 'second' not in args.pretrained_model:
         # As JEPA architecture includes encoder, target_encoder and predictor, load encoder only from JEPA
        model.load_ad_l_jepa_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)
    elif args.pretrained_model is not None and 'ad_l_jepa' not in args.pretrained_model:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)
    if args.ckpt:
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.extract_knn_one_epoch(
        cfg, model, data_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file, training=training
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None

def main():
    set_random_seed(1024)
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if args.ckpt:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    elif args.pretrained_model:
        num_list = re.findall(r'\d+', args.pretrained_model) if args.pretrained_model is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    if not args.ckpt and not args.pretrained_model:
        epoch_id = 0
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / 'eval_ssl'
    
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    if not os.path.exists(eval_output_dir / 'train_features.pt') or not os.path.exists(eval_output_dir / 'train_labels.pt'):
        logger.info('Extracting training set features and labels')
        # extract training set feature and labels
        train_set, train_loader, train_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=True
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
        with torch.no_grad():
            eval_single_ckpt(model, train_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test, training=True)
    else:
        logger.info('Training set features and labels already extracted')
        
    if not os.path.exists(eval_output_dir / 'val_features.pt') or not os.path.exists(eval_output_dir / 'val_labels.pt'):
        logger.info('Extracting validation set features and labels')
        # extract validation set feature and labels
        val_set, val_loader, val_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_set)
        with torch.no_grad():
            eval_single_ckpt(model, val_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test, training=False)
    else:
        logger.info('Validation set features and labels already extracted')
        
    if cfg.LOCAL_RANK == 0:
        train_features = torch.load(eval_output_dir / 'train_features.pt').to('cuda:0')
        train_labels = torch.load(eval_output_dir / 'train_labels.pt')
        val_features = torch.load(eval_output_dir / 'val_features.pt').to('cuda:0')
        val_labels = torch.load(eval_output_dir / 'val_labels.pt')
        for k in [20]:
            top1_class_accuracy, average_accuracy, predicted_top1_class, total_class = eval_utils.knn_classifier(train_features, train_labels,
               val_features, val_labels, k, T=0.07, num_classes=len(cfg.CLASS_NAMES))
            print(f"{k}-NN classifier result: Top1 per class: {top1_class_accuracy}, Average accuracy over classes: {average_accuracy}")
        
        res_dict = {
            'top1_class_accuracy': top1_class_accuracy,
            'average_accuracy': average_accuracy,
            'predicted_top1_class': predicted_top1_class,
            'total_class': total_class
        }
        # Save the results to a JSON file
        json_file = eval_output_dir / 'knn_results.json'
        with open(json_file, 'w') as f:
            json.dump(res_dict, f)

if __name__ == '__main__':
    main()
