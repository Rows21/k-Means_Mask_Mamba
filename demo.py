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
import torchvision
from load_datasets_transforms import SSLDataset, LAHeart, data_loader, data_transforms, infer_post_transforms
from load_datasets_transforms import Resize

import os
import tqdm
import numpy as np
import argparse

import math
import h5py

import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/mnt/3DUXNET/KMax-Net/demo/mri_norm2.h5', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/mnt/3DUXNET/KMax-Net/demo/', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='la', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='DKUNET', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='/mnt/3DUXNET/KMax-Net/outputs/LA_pretrain_20.pth', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=1, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

set_determinism(seed=0)

test_transforms = data_transforms(args)
test_samples, out_classes = data_loader(args)

def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = sliding_window_inference(test_patch, patch_size, 4, model)
                    y = torch.sigmoid(y[0])
                y = y.cpu().data.numpy()
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.uint8)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

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

h5f = h5py.File(args.root, 'r')
image = h5f['image'][:]
label = h5f['label'][:]
item = {'image': image, 'mask': label}

if args.dataset == 'la':        
    roi_size = (112, 112, 80)

transforms = torchvision.transforms.Compose([
                Resize(roi_size),
            ])
#prediction, score_map = test_single_case_first_output(model, image, stride_xy=18, stride_z=4, patch_size=(112,112,80), num_classes=2)
images = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
item = transforms(item)
item = {key: torch.from_numpy(value).unsqueeze(0).unsqueeze(0) for key, value in item.items()}
prediction = sliding_window_inference(images, (112, 112, 80), 1, model)

img_itk = sitk.GetImageFromArray(image)
prd_itk = sitk.GetImageFromArray(prediction[0])
lab_itk = sitk.GetImageFromArray(label)

case = args.dataset
test_save_path = args.output
sitk.WriteImage(prd_itk, test_save_path + case + '_20_4' + "_pred.nii.gz")
sitk.WriteImage(img_itk, test_save_path + case + '_20_4' + "_img.nii.gz")
sitk.WriteImage(lab_itk, test_save_path + case + '_20_4' + "_gt.nii.gz")


'''
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
        test_outputs_list = decollate_batch(test_outputs)
        test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]
        #y_pred = torch.stack(test_output_convert, dim=0).cpu().numpy()
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

    mean_dice_test = np.mean(dice_test)
    hd_test = np.mean(hd_test)
    asd_test = np.mean(asd_test)
    ji_test = np.mean(ji_test)
print(mean_dice_test,ji_test,hd_test,asd_test)
'''