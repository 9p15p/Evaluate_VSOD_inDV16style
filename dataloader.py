from torch.utils import data
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
# import time
# from tqdm import tqdm

class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root, use_flow):
        self.use_flow = use_flow
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        for name in lst_label:
            if name in lst_pred:
                lst.append(name)
        self.image_path = self.get_paths(lst, img_root)
        self.label_path = self.get_paths(lst, label_root)
        self.key_list = list(self.image_path.keys())

        self.check_path(self.image_path, self.label_path)
        self.trans = transforms.Compose([transforms.ToTensor()])


    def check_path(self, image_path_dict, label_path_dict):
        assert image_path_dict.keys() == label_path_dict.keys(), 'gt, pred must have the same videos'
        for k in image_path_dict.keys():
            assert len(image_path_dict[k]) == len(image_path_dict[k]), f'{k} have different frames'

    def get_paths(self, lst, root):
        v_lst = list(map(lambda x: os.path.join(root, x), lst))

        f_lst = {}
        for v in v_lst:
            v_name = v.split('/')[-1]
            if 'result' in root:
                if not self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:]
                elif self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:-1]  # 光流方法忽略第一帧和最后一帧

            elif 'gt' in root:
                if not self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:]
                elif self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:-1]  # 光流方法忽略第一帧和最后一帧
        return f_lst

    def read_picts(self, v_name):
        pred_names = self.image_path[v_name]
        pred_list = []
        for pred_n in pred_names:
            pred_list.append(self.trans(Image.open(pred_n).convert('L')))

        gt_names = self.label_path[v_name]
        gt_list = []
        for gt_n in gt_names:
            gt_list.append(self.trans(Image.open(gt_n).convert('L')))

        for gt, pred in zip(gt_list, pred_list):
            assert gt.shape == pred.shape, 'gt.shape!=pred.shape'
        
        gt_list = torch.cat(gt_list,dim=0)
        pred_list = torch.cat(pred_list,dim=0)
        return pred_list, gt_list

    def __getitem__(self, item):
        v_name = self.key_list[item]
        preds, gts = self.read_picts(v_name)

        return v_name, preds, gts

    def __len__(self):
        return len(self.image_path)

# if __name__ == '__main__':
#     img_root = '../result/fsnet/DAVIS/'
#     label_root = '../../dataset/gt/DAVIS/'
#     use_flow = False
#     dataset = EvalDataset(img_root, label_root, use_flow)
#     time1 = time.time()
#     for v_name, preds, gts in tqdm(dataset):
#         pass
#     time2 = time.time()
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=1,
#                                   shuffle=False,
#                                   num_workers=12,
#                                   pin_memory=True,
#                                   drop_last=False)
#     for i, batch in enumerate(data_loader): 
#         pass
#     time3 = time.time()
#     print('tqdm', time2-time1)
#     print('dataloader', time3-time2)
