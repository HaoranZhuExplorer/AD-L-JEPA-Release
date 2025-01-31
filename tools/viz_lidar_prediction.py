import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
    print("using open3d")
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
    print("using mayavi")

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    logger.info(cfg.DATA_CONFIG)
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, logger=logger, training=False
    )


    logger.info(f'Total number of samples: \t{len(train_set)}')
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(train_set):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = train_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'][pred_dicts[0]['pred_scores']>0.4]
            pred_scores = pred_dicts[0]['pred_scores'][pred_dicts[0]['pred_scores']>0.4]
            pred_labels = pred_dicts[0]['pred_labels'][pred_dicts[0]['pred_scores']>0.4]
            print("debug frame id", data_dict['frame_id'])
            print("debug gt_boxes numbers:", data_dict['gt_boxes'].shape[1], "gt_boxes labels:", data_dict['gt_boxes'][0, :, 7].cpu().numpy())
            print("debug predicted boxes (with confidence score>0.4) numbers:", pred_boxes.shape[0], "predicted boxes labels:", pred_labels.cpu().numpy())
            
            points = data_dict['points'].cpu().numpy()
            # gt boxes: red; pred: white, green, cyan, yellow
            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0, :, :8], ref_boxes=pred_boxes,
                ref_scores=pred_scores, ref_labels=pred_labels
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
            
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
