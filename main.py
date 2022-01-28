import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset


# from concurrent.futures import ThreadPoolExecutor
def Eval_VSOD(cfg):
    if cfg.methods is None:
        method_names = os.listdir(cfg.pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(cfg.gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')

    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(pred_root=osp.join(cfg.pred_dir, method, dataset),
                                 label_root=osp.join(cfg.gt_dir, dataset),
                                 use_flow=config.use_flow)
            thread = Eval_thread(loader, method, dataset, cfg.log_dir)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第一帧与最后一帧若--use_flow则不eval")
    parser.add_argument('--methods', type=str, default=None, help='字符串格式，算法名称')
    parser.add_argument('--datasets', type=str, default=None, help='验证的数据集，数据集之间用空格隔开')
    parser.add_argument('--gt_dir', type=str, default='./gt', help='e.g. DAVIS2016的上一层文件夹')
    parser.add_argument('--pred_dir', type=str, default='./pred', help='e.g. LWL4vsod\/DAVIS2016的上一层文件夹')
    parser.add_argument('--log_dir', type=str, default='./', help='保存log的位置')
    parser.add_argument('--use_flow', type=bool, default=False,help="如果使用光流则在第一帧和最后一帧GT上不eval【具体调整见dataloader】")

    config = parser.parse_args()
    Eval_VSOD(config)
