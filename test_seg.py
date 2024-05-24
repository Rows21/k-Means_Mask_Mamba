from monai.utils import first, set_determinism

from monai.networks import one_hot
from monai.transforms import AsDiscrete
from networks.DKUNet.network_backbone import DKUNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

from iou import MeanIoU
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from medpy import metric
import torch
from dataloader import SSLDataset, LAHeart, data_loader, data_transforms, infer_post_transforms

import os
import tqdm
import SimpleITK as sitk
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/mnt/3DUXNET/3DUX-Net/datasets/LA2018', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/mnt/3DUXNET/KMax-Net/demo/', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='la', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='DKUNET', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='/mnt/3DUXNET/KMax-Net/outputs/LA_pretrain_10.pth', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=1, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

set_determinism(seed=0)

test_transforms = data_transforms(args)

## Inference Pytorch Data Loader and Caching
if args.dataset == 'la':
    test_samples, out_classes = data_loader(args)
    test_ds = LAHeart(base_dir=args.root,
                           split='test',
                           transform_val=test_transforms,
                           with_idx=True)
    
    test_loader = DataLoader(test_ds, 
                            batch_size=args.sw_batch_size, 
                            num_workers=args.num_workers)

## Load Networks
device = torch.device("cuda:0")
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
model.load_state_dict(torch.load(args.trained_weights))

if args.dataset == 'la':        
    roi_size = (112, 112, 80)

post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
hd_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean",percentile=95)
asd_metric = SurfaceDistanceMetric(include_background=False)
ji_metric = MeanIoU(include_background=False)
model.eval()
hd_test = list()
dice_test = list()
asd_test = list()
ji_test = list()
with torch.no_grad():
    for k, test_data in enumerate(test_loader):
        print(k)
        images = test_data["image"].to(device)
        test_labels_list = decollate_batch(test_data["label"].to(device))
        test_labels_convert = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
        #y = torch.stack(test_labels_convert, dim=0).cpu().numpy()
        
        test_outputs = sliding_window_inference(images, (112, 112, 80), 4, model)
        test_outputs = test_outputs[0]

        images = images.squeeze().cpu().numpy()
        labels = test_data["label"].squeeze().cpu().numpy()

        img_itk = sitk.GetImageFromArray(images)
        lab_itk = sitk.GetImageFromArray(labels)

        case = args.dataset
        test_save_path = args.output
        ablation = str(k) + '_10_4'
       
        sitk.WriteImage(img_itk, test_save_path + case + ablation + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + ablation + "_gt.nii.gz")

        test_outputs_list = decollate_batch(test_outputs)
        test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]
        prd_itk = sitk.GetImageFromArray(test_output_convert[0][1,:,:,:].cpu().numpy())
        sitk.WriteImage(prd_itk, test_save_path + case + ablation + "_pred.nii.gz")
        y_pred = torch.stack(test_output_convert, dim=0).cpu().numpy()
        dice_metric(y_pred=test_output_convert, y=test_labels_convert)
        dice = dice_metric.aggregate().item()
        dice_test.append(dice)

        hd_metric(y_pred=test_output_convert, y=test_labels_convert)
        hd = hd_metric.aggregate().item()
        hd_test.append(hd)

        asd_metric(y_pred=test_output_convert, y=test_labels_convert)
        asd = asd_metric.aggregate().item()
        asd_test.append(asd)

        ji_metric(y_pred=test_output_convert, y=test_labels_convert)
        ji = ji_metric.aggregate().item()
        ji_test.append(ji)
        print(dice)
    mean_dice_test = np.mean(dice_test)
    hd_test = np.mean(hd_test)
    asd_test = np.mean(asd_test)
    ji_test = np.mean(ji_test)
print(mean_dice_test,ji_test,hd_test,asd_test)