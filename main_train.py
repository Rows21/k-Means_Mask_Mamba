#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Row21
"""

import warnings

import torch
import torch.nn.functional as FF

from monai.utils import set_determinism
from monai.transforms import AsDiscrete

from networks.DKUNet.network_backbone import DKUNET
from networks.DKUNet.criterion import SetCriterion
from monai.networks.nets import UNETR, SwinUNETR
from monai.networks import one_hot
# from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

import torch
import torch.nn as nn
import warnings
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import SSLDataset, data_loader, data_transforms, LAHeart

import os
import numpy as np
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser(description='DK-UNet hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/mnt/datasets/LA2018', help='')
parser.add_argument('--output', type=str, default='/mnt/3DUXNET/KMax-Net/outputs', help='')
parser.add_argument('--dataset', type=str, default='la', help='Datasets: {la, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--labeled_ratio', type=int, default=20, help='Labeled Ratio for Semi Supervised Learning')

## Input model & training hyperparameters
parser.add_argument("--local_rank", type=int)
parser.add_argument('--network', type=str, default='DKUNET', help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, DKUNET}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='/mnt/3DUXNET/KMax-Net/pretrain/LA_pretrain_5.pth', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=2000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=800, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1.0, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()
print("torch availability:",torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

set_determinism(seed=0)

train_strong_transforms, train_weak_transforms, train_transforms, val_transforms = data_transforms(args)
print('Start caching datasets!')

labeled_ratio=args.labeled_ratio

if args.dataset == 'la':
    train_samples, valid_samples, out_classes = data_loader(args)
    train_ds = LAHeart(base_dir=args.root,
                           split='train',
                           transform_strong=train_strong_transforms,
                           transform_weak=train_weak_transforms,
                           with_idx=True)
    val_ds = LAHeart(base_dir=args.root,
                           split='test',
                           transform_val=val_transforms,
                           with_idx=True)
    train_loader = DataLoader(train_ds, 
                            batch_size=args.batch_size * 2, 
                            shuffle=True, 
                            #num_workers=args.num_workers, 
                            pin_memory=True)
    val_loader = DataLoader(val_ds, 
                            batch_size=1, 
                            num_workers=args.num_workers)
    
    train_files = train_ds.image_list
    split_idx = int(len(train_ds.image_list) * labeled_ratio/100/2)
    full_list = list(range(int(len(train_ds.image_list)/2)))
    
else:
    train_samples, valid_samples, out_classes = data_loader(args)

    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
    ]

    val_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
    ]

    split_idx = int(len(train_files) * labeled_ratio/100)
    full_list = list(range(len(train_files)))

    #for i in unlabeled_list:
    #    train_files[i]['label'] = [-1]

    ## Train Pytorch Data Loader and Caching
    train_ds = SSLDataset(data=train_files,
                            transform_weak=train_strong_transforms, #train_strong_transforms, 
                            transform_strong=train_weak_transforms, #train_weak_transforms
                            #unlabeled_list=unlabeled_list
                            )
    ## Valid Pytorch Data Loader and Caching
    val_ds = CacheDataset(data=val_files, 
                            transform=val_transforms, 
                            cache_rate=args.cache_rate, 
                            num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers, 
                            pin_memory=True)
    val_loader = DataLoader(val_ds, 
                            batch_size=1, 
                            num_workers=args.num_workers)

## Set up unlabeled list
random.shuffle(full_list)
labeled_list = full_list[:split_idx]
unlabeled_list = full_list[split_idx:]
print('Labeled Ratio:{}%'.format(labeled_ratio))
print('Chosen Labeled idx: {}'.format(labeled_list))
## Load Networks
# torch.cuda.set_device(args.local_rank)
# torch.distributed.init_process_group(backend='nccl')
device = torch.device("cuda")
# device = torch.device("cpu")
if args.network == 'DKUNET':
    model = DKUNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).cuda()# .to(device)
    model = torch.nn.DataParallel(model)
        
        #model.load_state_dict(torch.load(args.pretrained_weights), strict = False) # strict = False
print('Chosen Network Architecture: {}'.format(args.network))

if args.pretrain == True:
        print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
        model.load_state_dict(torch.load(args.pretrained_weights), strict = False)

## Define Loss function and optimizer
# seg_loss = DiceCELoss(to_onehot_y=True, softmax=True)
class SetCriterion(nn.Module):
    """
    Args:
        input: the shape should be BNH[WD].
        target: the shape should be BNH[WD] or B1H[WD].

    Raises:
        ValueError: When number of dimensions for input and target are different.
        ValueError: When number of channels for target is neither 1 nor the same as input.

    """
    def __init__(self,
            to_onehot_y=True,
            softmax=True,
            data_size=None,
            nce = 'Cosine'
        ) -> None:
            super().__init__()
            self.seg_loss = DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax)
            self.to_onehot_y = to_onehot_y
            self.lambda_qd = 0.1
            self.lambda_consistency = 0.05
            self.lambda_ce = 1.0
            self.temperature = 0.5
            self.reg_radius: float = 200
            self.reg_coef: float = 0
            self.data_size = data_size
            self.nce = nce

    def InfoNCECosine(self,a,b):
        # mean deviation from the sphere with radius `reg_radius`
        batch_size,_,H,W,D = a.size()
        vecnorms = torch.linalg.vector_norm(torch.cat((a,b),dim=0), dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * FF.mse_loss(vecnorms, target)

        a = FF.normalize(a).flatten(1) # B x (KL)
        b = FF.normalize(b).flatten(1) # B x (KL)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss/(H*W*D)
    
    def InfoNCECauchy(self,a,b):
        # backbone_features and labels are unused
        batch_size = a.size()[0]
        a = FF.normalize(a).flatten(1)
        b = FF.normalize(b).flatten(1)

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diagonal_copy(sim_ab).log_().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss
    
    def InfoNCEGaussian(self,a,b):
        # backbone_features and labels are unused
        batch_size = a.size()[0]
        a = FF.normalize(a).flatten(1)
        b = FF.normalize(b).flatten(1)

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss
    
    def qd_loss(self,x,y):
            # x : B x 3 x H x W x D
            # y : B x K x H x W x D
            n_pred_ch = x.shape[1]

            # y : B x K x H x W x D -> B x 3 x H x W x D
            y[y > 2] = 2
            if self.to_onehot_y:
                if n_pred_ch == 1:
                    warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
                else:
                    y = one_hot(y, num_classes=n_pred_ch)

            x = x.flatten(2) # B x 3 x L
            y = y.flatten(2) # B x 3 x L

            qd = - torch.mean(y * torch.log(x))
            return qd
        
    def forward(self, 
                logit_map_s: torch.Tensor, logit_map_w: torch.Tensor, 
                qd_strong: torch.Tensor, qd_weak: torch.Tensor, 
                target_s: torch.Tensor = None, target_w: torch.Tensor = None,
                loss_step:int = None) -> torch.Tensor:
            """
            Args:
                logit_map: the shape should be BNH[WD].
                target: the shape should be BNH[WD] or B1H[WD].

            Raises:
                ValueError: When number of dimensions for input and target are different.
                ValueError: When number of channels for target is neither 1 nor the same as input.

            """
            
            if loss_step <= self.data_size *5:
                self.lambda_consistency = 0.1
            elif loss_step > self.data_size *5 and loss_step <= self.data_size *20:
                self.lambda_consistency = 0.5
            elif loss_step > self.data_size *20 and loss_step <= self.data_size *40:
                self.lambda_consistency = 1.0
            elif loss_step > self.data_size*40 and loss_step <= self.data_size*80:
                self.lambda_consistency = 2.0
                   
            # InfoNCE loss
            if self.nce == 'Cosine':
                seg_consistency_loss = self.InfoNCECosine(logit_map_s, logit_map_w)
                #seg_consistency_loss = 0
                query_consistency_loss = self.InfoNCECosine(qd_strong, qd_weak)
                #query_consistency_loss = 0
            elif self.nce == 'Cauchy':
                seg_consistency_loss = self.InfoNCECauchy(logit_map_s, logit_map_w)
                query_consistency_loss = self.InfoNCECauchy(qd_strong, qd_weak)
            elif self.nce == 'Gaussian':
                seg_consistency_loss = self.InfoNCEGaussian(logit_map_s, logit_map_w)
                query_consistency_loss = self.InfoNCEGaussian(qd_strong, qd_weak)

            if target_s is not None:
                if len(logit_map_s.shape) != len(target_s.shape):
                    raise ValueError("the number of dimensions for input and target should be the same.")
                
                # segmentation loss
                seg_loss_s = self.seg_loss(logit_map_s, target_s)
                seg_loss_w = self.seg_loss(logit_map_w, target_w)
                seg_loss = seg_loss_s + seg_loss_w
                
                # qd loss
                dice_loss = self.qd_loss(qd_strong, target_s) + self.qd_loss(qd_weak, target_w)
            else:
                dice_loss = 0
                seg_loss = 0

            total_loss: torch.Tensor = self.lambda_ce * seg_loss + self.lambda_qd * dice_loss #+ self.lambda_consistency * query_consistency_loss #+ seg_consistency_loss)
            #total_loss: torch.Tensor = self.lambda_ce * seg_loss + self.lambda_consistency * dice_loss

            return total_loss

criterion = SetCriterion(data_size=len(full_list))
    
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)

root_dir = os.path.join(args.output)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)
        
t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
        # model_feat.eval()
        model.eval()
        dice_vals = list()
        hd_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                #val_outputs,_ = model(val_inputs)
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model)
                val_outputs = val_outputs[0]
                # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)

                hd_metric(y_pred=val_output_convert, y=val_labels_convert)
                hd = hd_metric.aggregate().item()
                hd_vals.append(hd)

                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f, hd=%2.5f)" % (step+1, 20.0, dice,hd)
                )
            dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        mean_hd = np.mean(dice_vals)
        writer.add_scalar('Validation Segmentation Loss', mean_dice_val,mean_hd,global_step)
        return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
        # model_feat.eval()
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):

            #epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            #dice_val = validation(epoch_iterator_val)
            step += 1
            if args.dataset == 'la':
                 _,_,_,H,W,D = batch['image'].size()
                 # reshape batch
                 batch['image'] = torch.reshape(batch['image'], (4, 1, H,W,D))
                 batch['image'] = torch.index_select(batch['image'], 0, torch.tensor([0,2,1,3]))

                 batch['label'] = torch.reshape(batch['label'], (4, 1, H,W,D))
                 batch['label'] = torch.index_select(batch['label'], 0, torch.tensor([0,2,1,3]))

            x, y = (batch["image"].cuda(), batch["label"].cuda())

            if labeled_ratio != 0:
                if step % len(train_files) in unlabeled_list:
                    global_step += 1
                    continue
                    y_strong = None
                    y_weak = None
                else:
                    y_strong, y_weak = torch.chunk(y, 2, 0)
            else:
                y_strong, y_weak = torch.chunk(y, 2, 0)
                 
            
            if x.dtype != torch.float32:
                 x = x.float()
            
            logit_map = model(x)
            
            seg,qd = logit_map
            seg_strong, seg_weak = torch.chunk(seg,2,0)
            qd_strong, qd_weak = torch.chunk(qd,2,0)
            # weak
            # seg_weak, qd_weak = model(batch_weak["image"])

            loss = criterion(seg_strong, seg_weak,
                             qd_strong, qd_weak,
                             y_strong, y_weak, step)

            
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            if (
                global_step % eval_num == 0 and global_step != 0 # eval_num
            ) or global_step == (max_iterations - 1):
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, args.dataset + "best_metric_model.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                    # scheduler.step(dice_val)
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                    # scheduler.step(dice_val)
            writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
            if step == global_step/10:
                torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                    )
            global_step += 1
            torch.cuda.empty_cache()
        return global_step, dice_val_best, global_step_best

if __name__ == '__main__': 
    max_iterations = args.max_iter
    print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
    eval_num = args.eval_step
    post_label = AsDiscrete(to_onehot=out_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean",percentile=95)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )