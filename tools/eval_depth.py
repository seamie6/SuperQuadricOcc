import numpy as np
import torch
import re
import argparse
import os

import torch.nn.functional as F

class DepthEvaluator(object):
    def __init__(self,
            pred_dir='',
            min_depth=0.1,
            max_depth=80):

        if not os.path.exists(pred_dir):
            raise ValueError(f'Predicted depth root {pred_dir} does not exist.')
            
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pred_dir = pred_dir

        self.gt_depth_root = 'data/gt_depth/samples'
        self.pkl_file = 'data/bevdetv2-nuscenes_infos_val.pkl'

        self.cams = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
        ]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def compute_errors(self, gt, pred):
        thresh = torch.maximum(gt / pred, pred / gt)

        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        rmse = torch.sqrt(((gt - pred) ** 2).mean())
        rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean())

        abs_rel = (torch.abs(gt - pred) / gt).mean()
        sq_rel = (((gt - pred) ** 2) / gt).mean()

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def evaluate(self):
        data_pkl = np.load(self.pkl_file, allow_pickle=True)
        infos = data_pkl['infos']

        abs_rel_all = []
        sq_rel_all = []
        rmse_all = []
        rmse_log_all = []
        a1_all = []
        a2_all = []
        a3_all = []

        for k, info in enumerate(infos):
            token = info['token']
            pred_depth_path = os.path.join(self.pred_dir, f'{token}.npz')

            gt_depth_all = []

            for cam in self.cams:
                cam_info = infos[k]['cams'][cam]
                data_path = cam_info['data_path']

                filename = re.search(r'.*/(n\d{3}-[^/]+)\.jpg', data_path).group(1)

                gt_depth_cam_path = os.path.join(self.gt_depth_root, cam, f'{filename}.npy')

                gt_depth = np.load(gt_depth_cam_path)
                gt_depth_all.append(torch.from_numpy(gt_depth))

            gt_depth_all = torch.stack(gt_depth_all, dim=0).to(self.device, non_blocking=True)
            pred_depth = np.load(pred_depth_path)['arr_0']
            pred_depth = torch.from_numpy(pred_depth).to(self.device, non_blocking=True)

            pred_depth = pred_depth.unsqueeze(1).float()
            pred_depth = F.interpolate(
                pred_depth,
                size=(900, 1600),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            mask = (gt_depth_all > self.min_depth) & (gt_depth_all < self.max_depth)
            gt = gt_depth_all[mask]
            pred = pred_depth[mask].clamp(self.min_depth, self.max_depth)

            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.compute_errors(
                gt, pred)
            
            abs_rel_all.append(abs_rel.item())
            sq_rel_all.append(sq_rel.item())
            rmse_all.append(rmse.item())
            rmse_log_all.append(rmse_log.item())
            a1_all.append(a1.item())
            a2_all.append(a2.item())
            a3_all.append(a3.item())

            if k%100==0:
                print(f'======= Values at current iteration {k} =======')
                print("abs_rel: ", np.mean(abs_rel_all))
                print("sq_rel: ", np.mean(sq_rel_all))
                print("rmse: ", np.mean(rmse_all))
                print("rmse_log: ", np.mean(rmse_log_all))
                print("a1: ", np.mean(a1_all))
                print("a2: ", np.mean(a2_all))
                print("a3: ", np.mean(a3_all))

        print(f'======= Final values =======')
        print("abs_rel: ", np.mean(abs_rel_all))
        print("sq_rel: ", np.mean(sq_rel_all))
        print("rmse: ", np.mean(rmse_all))
        print("rmse_log: ", np.mean(rmse_log_all))
        print("a1: ", np.mean(a1_all))
        print("a2: ", np.mean(a2_all))
        print("a3: ", np.mean(a3_all))

    def __call__(self):
        return self.evaluate()
    
def parse_args():
    parser = argparse.ArgumentParser(description='Depth evaluation')

    parser.add_argument(
        '--pred-dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--min-depth',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--max-depth',
        type=float,
        default=80.0
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = DepthEvaluator(
        pred_dir=args.pred_dir,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    model()