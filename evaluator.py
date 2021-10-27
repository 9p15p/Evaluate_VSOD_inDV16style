import os
import time

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import csv


class Eval_thread():
    def __init__(self, loader, method, dataset, output_dir, cuda):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.logfile = os.path.join(output_dir, f'{dataset}.csv')

    def run(self):
        start_time = time.time()
        mae = self.Eval_mae()
        max_f = self.Eval_fmeasure()
        max_e = self.Eval_Emeasure()
        s = self.Eval_Smeasure()
        self.LOG(
            {"Method": self.method, "MAE": mae, "F-Max-measure": max_f, "E-Max-measure": max_e, "S-measure": s}
        )
        return '[cost:{:.4f}s]{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure, {:.4f} S-measure..'.format(
            time.time() - start_time, self.dataset, self.method, mae, max_f, max_e, s)

    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        mae_dict = dict()
        with torch.no_grad():
            for v_name, preds, gts in tqdm(self.loader):
                preds = preds.cuda() if self.cuda else preds
                gts = gts.cuda() if self.cuda else gts

                mean = torch.abs(preds - gts).mean()
                assert mean == mean, "mean is NaN"  # for Nan
                mae_dict[v_name] = mean
            # 所有视频求平均
            maE_videos_max = torch.mean(torch.tensor(list(mae_dict.values())))
            print(F"maE_videos_max: {maE_videos_max}")
            return maE_videos_max

    def Eval_fmeasure(self):
        print('eval[FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        F_dict = dict()
        with torch.no_grad():
            for v_name, preds, gts in tqdm(self.loader):
                preds = preds.cuda() if self.cuda else preds
                gts = gts.cuda() if self.cuda else gts
                f_score = 0
                for pred, gt in zip(preds, gts):
                    prec, recall = self._eval_pr(pred, gt, 255)
                    f_score += (1 + beta2) * prec * recall / (beta2 * prec + recall)
                    f_score[f_score != f_score] = 0  # for Nan
                    assert (f_score == f_score).all()  # for Nan
                f_score /= len(preds)
                # 单个视频的F
                F_dict[v_name] = f_score

            # 所有视频的
            F_videos = torch.stack(list(F_dict.values())).mean(dim=0)
            F_videos_max = F_videos.max()
            
            print(f'F_videos_max:{F_videos_max}')
            return F_videos_max

    def Eval_Emeasure(self):
        print('eval[EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        E_dict = dict()
        with torch.no_grad():
            for v_name, preds, gts in tqdm(self.loader):
                e_score = torch.zeros(255).cuda() if self.cuda else torch.zeros(255)
                preds = preds.cuda() if self.cuda else preds
                gts = gts.cuda() if self.cuda else gts
                for pred, gt in zip(preds, gts):
                    e_score += self._eval_e(pred, gt, 255)
                e_score /= len(preds)
                # 单个视频的E
                E_dict[v_name] = e_score

            # 所有视频的
            E_videos = torch.stack(list(E_dict.values())).mean(dim=0)
            E_videos_max = E_videos.max()
            print(f'E_videos_max:{E_videos_max}')
            return E_videos_max

    def Eval_Smeasure(self):
        print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        alpha = 0.5
        S_dict = dict()
        with torch.no_grad():
            for v_name, preds, gts in tqdm(self.loader):
                preds = preds.cuda() if self.cuda else preds
                gts = gts.cuda() if self.cuda else gts
                sum_Q = 0
                for pred, gt in zip(preds, gts):
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    sum_Q += Q

                # 单个视频的S
                S_video = sum_Q / len(preds)
                S_dict[v_name] = S_video
            # 所有视频的
            S_videos_mean = torch.mean(torch.tensor(list(S_dict.values())))
            return S_videos_mean

    def LOG(self, output):
        mode = 'a+' if os.path.exists(self.logfile) else 'w'
        headers = output.keys()
        with open(self.logfile, mode, encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, headers)
            if mode == 'w':
                writer.writeheader()
            writer.writerow(output)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q


if __name__ == '__main__':
    pass
    # [cost:278.0082s]DAVIS2016 dataset with 1_25 method get 0.0203 mae, 0.8465 max-fmeasure, 0.9528 max-Emeasure, 0.8797 S-measure..

    # def writer_csv_demo2():
    #     headers = ["name", "age", "height"]
    #     values = [
    #         {"name": "小王", "age": 18, "height": 178},
    #         {"name": "小王", "age": 18, "height": 178},
    #         {"name": "小王", "age": 18, "height": 178}
    #     ]
    #     flag = os.path.exists("classromm2.csv")
    #     with open("classromm2.csv", "a+", encoding="utf-8", newline="") as fp:
    #         writer = csv.DictWriter(fp, headers)  # 使用csv.DictWriter()方法，需传入两个个参数，第一个为对象，第二个为文件的title
    #         if not flag:
    #             writer.writeheader()  # 使用此方法，写入表头
    #         writer.writerows(values)
    #
    # writer_csv_demo2()

    # def LOG(logfile, output):
    #     mode = 'a+' if os.path.exists(logfile) else 'w'
    #     headers = output.keys()
    #     with open(logfile, mode, encoding='utf-8', newline='') as f:
    #         writer = csv.DictWriter(f, headers)
    #         if mode == 'w':
    #             writer.writeheader()
    #         writer.writerow(output)
    # output = {"Method": "1_25", "MAE": 0.0203, "F-Max-measure": 0.8465, "E-Max-measure": 0.9528, "S-measure": 0.8797}
    # LOG('results.csv', output)
