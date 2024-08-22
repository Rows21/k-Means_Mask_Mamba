from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
from copy import deepcopy
from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    HistogramNormalize,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous

import volumentations
import torchvision
import torch
import numpy as np

import random
import glob
from monai.data import Dataset

import sys

import warnings
from copy import copy, deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL
from tqdm import tqdm
import h5py

from monai.data import CacheDataset, DataLoader
from monai.data.utils import pickle_hashing
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous
    
def data_loader(args):
    root_dir = args.root
    dataset = args.dataset

    print('Start to load data from directory: {}'.format(root_dir))

    if dataset == 'la':
        out_classes = 2

        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}

            ## Input training data
            train_samples = sorted(glob.glob(os.path.join(root_dir,'Training Set/*','mri_norm2.h5')))
            #train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
            #train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
            #train_samples['images'] = train_img
            #train_samples['labels'] = train_label

            valid_samples = sorted(glob.glob(os.path.join(root_dir,'Testing Set/*','mri_norm2.h5')))

            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            print('Number of classes for segmentation: {}'.format(out_classes))

            return train_samples, valid_samples, out_classes

        elif args.mode == 'test':
            test_samples = {}

            ## Input inference data
            test_samples = sorted(glob.glob(os.path.join(root_dir,'Training Set/*','mri_norm2.h5')))
            print('Finished loading all training samples from dataset: {}!'.format(dataset))


            return test_samples, out_classes

    if dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16

    if args.mode == 'train':
        train_samples = {}
        valid_samples = {}

        ## Input training data
        train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
        train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        train_samples['images'] = train_img
        train_samples['labels'] = train_label

        ## Input validation data
        valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
        valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        ## Input inference data
        test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        test_samples['images'] = test_img

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes

def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset == 'flare':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96), # 96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        train_strong_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96), # 96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                #RandShiftIntensityd(
                #    keys=["image"],
                #    offsets=0.10,
                #    prob=0.50,
                #),
                #RandAffined(
                #    keys=['image', 'label'],
                #    mode=('bilinear', 'nearest'),
                #    prob=1.0, spatial_size=(96, 96, 96),
                #    rotate_range=(0, 0, np.pi / 30),
                #    scale_range=(0.1, 0.1, 0.1)),
                RandGaussianNoised(
                    keys=['image', 'label'],
                    prob=0.5, # 0.5
                    mean=0.0,
                    std=0.1
                ),
                RandAdjustContrastd(
                    keys=['image', 'label'],
                    prob=0.5, # 0.5
                    gamma=(0.5, 4.5)
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        train_weak_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96), # 96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                #RandShiftIntensityd(
                #    keys=["image"],
                #    offsets=0.10,
                #    prob=0.50,
                #),
                #RandAffined(
                #    keys=['image', 'label'],
                #    mode=('bilinear', 'nearest'),
                #    prob=1.0, spatial_size=(96, 96, 96),
                #    rotate_range=(0, 0, np.pi / 30),
                #    scale_range=(0.1, 0.1, 0.1)),
                RandGaussianNoised(
                    keys=['image', 'label'],
                    prob=0.5, # 0.5
                    mean=0.0,
                    std=0.05
                ),
                RandCoarseDropoutd(
                    keys='image',
                    prob=0.5, # 0.5
                    holes=100,
                    spatial_size=10
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear")),
                # ResizeWithPadOrCropd(keys=["image"], spatial_size=(168,168,128), mode=("constant")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'amos':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        train_strong_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96), # 96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                #RandShiftIntensityd(
                #    keys=["image"],
                #    offsets=0.10,
                #    prob=0.50,
                #),
                #RandAffined(
                #    keys=['image', 'label'],
                #    mode=('bilinear', 'nearest'),
                #    prob=1.0, spatial_size=(96, 96, 96),
                #    rotate_range=(0, 0, np.pi / 30),
                #    scale_range=(0.1, 0.1, 0.1)),
                RandGaussianNoised(
                    keys=['image', 'label'],
                    prob=0.5, # 0.5
                    mean=0.0,
                    std=0.1
                ),
                RandAdjustContrastd(
                    keys=['image', 'label'],
                    prob=0.5, # 0.5
                    gamma=(0.5, 4.5)
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        train_weak_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96), # 96
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                #RandShiftIntensityd(
                #    keys=["image"],
                #    offsets=0.10,
                #    prob=0.50,
                #),
                #RandAffined(
                #    keys=['image', 'label'],
                #    mode=('bilinear', 'nearest'),
                #    prob=1.0, spatial_size=(96, 96, 96),
                #    rotate_range=(0, 0, np.pi / 30),
                #    scale_range=(0.1, 0.1, 0.1)),
                RandGaussianNoised(
                    keys=['image', 'label'],
                    prob=0.5, # 0.5
                    mean=0.0,
                    std=0.05
                ),
                RandCoarseDropoutd(
                    keys='image',
                    prob=0.5, # 0.5
                    holes=100,
                    spatial_size=10
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'la':
        output_size=(112,112,80)
        train_transforms = volumentations.Compose([
                volumentations.Resize((96,96,96), interpolation=1, resize_type=0, always_apply=True, p=1.0),
                volumentations.GaussianNoise(var_limit=(0, 5), p=0.2),
            ])

        train_strong_transforms = torchvision.transforms.Compose([
                RandomCrop(output_size),#volumentations.Resize((96,96,96), interpolation=1, resize_type=0, always_apply=True, p=1.0),
                GaussianNoise(var_limit=(0, 5)),#volumentations.GaussianNoise(var_limit=(0,5), p=0.5),
            ])

        train_weak_transforms = torchvision.transforms.Compose([
                RandomCrop(output_size),#volumentations.Resize((96,96,96), interpolation=1, resize_type=0, always_apply=True, p=1.0),
                #volumentations.GaussianNoise(var_limit=(0, 5), p=0.5),
            ])
        
        val_transforms = torchvision.transforms.Compose([
                RandomCrop(output_size),
            ])

        test_transforms = torchvision.transforms.Compose([
                #Resize(output_size),
            ])
    
    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_strong_transforms, train_weak_transforms, train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms

from skimage import transform as sk_trans
class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['mask']
        (w, h, d) = image.shape
        #label = label.astype(np.bool_)
        image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)
        assert (np.max(label) == 1 and np.min(label) == 0)
        assert (np.unique(label).shape[0] == 2)

        return {'image': image, 'mask': label}
    
class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = np.flip(img,1).copy()
        return img

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class GaussianNoise(object):
    def __init__(self, var_limit=(10.0, 50.0), mean=0, p=0.5):
        self.var_limit = var_limit
        self.mean = mean

    def __call__(self, sample):
        image, label = sample['image'], sample['mask']
        var = np.random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        gauss = np.random.normal(self.mean, sigma, image.shape).astype("float32")
        return {'image': image.astype("float32") + gauss, 'mask': label}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['mask']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'mask': label, 'sdf': sdf}
        else:
            return {'image': image, 'mask': label}
    
class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform_strong=None, transform_weak=None, transform_val=None, with_idx=False, as_contiguous=True):
        self._base_dir = base_dir
        self.transform_strong = transform_strong
        self.transform_weak = transform_weak
        self.sample_list = []
        self.with_idx = with_idx
        self.as_contiguous = as_contiguous
        self.split = split
        self.transform_val = transform_val

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/Training Set/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        item = {'image': image, 'mask': label}

        if self.split == 'train':
            # train strong for batched pics
            if self.transform_strong is not None:
                item_strong = self.transform_strong(item)
                item_strong = {key: torch.from_numpy(value).unsqueeze(0) for key, value in item_strong.items()}


            # train weak for batched pics
            if self.transform_strong is not None:
                item_weak = self.transform_weak(item)
                item_weak = {key: torch.from_numpy(value).unsqueeze(0) for key, value in item_weak.items()}
            self.sample_list = []
            [self.sample_list.append(b) for b in [item_strong,item_weak]]
            batch = self.collate_fn_dict(self.sample_list)
        
        elif self.split == 'test':
            item_val = self.transform_val(item)
            item_val = {key: torch.from_numpy(value).unsqueeze(0) for key, value in item_val.items()}
            batch = {'image': item_val['image'], 'label': item_val['mask']}

        if self.as_contiguous:
            batch = convert_to_contiguous(batch, memory_format=torch.contiguous_format)
        
        return batch
    
    def collate_fn_dict(self,batches):
        #[batch.extend(b) for b in batches]
        imgs = [s['image'].unsqueeze(0) for s in batches]
        imgs = torch.cat(imgs, dim=0)
        label = [s['mask'].unsqueeze(0) for s in batches]
        label = torch.cat(label, dim=0)
        return {'image': imgs, 'label': label}
    
class SSLDataset(Dataset):
    """Basic Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        roots (list): list of dirs of the dataset
        transform_post: transform object after cropping
        crop_fn: cropping function
        lesion_label (list): label names of lesion, such as ['Aneurysm']
    """

    def __init__(self, 
                 data, 
                 # transform_post = None, 
                 transform_strong = None, 
                 transform_weak = None, 
                 crop_fn = None, 
                 lesion_label = None, 
                 #unlabeled_list = None,
                 as_contiguous: bool = True):

        if lesion_label is None:
            self.lesion_label = [None]
        else:
            self.lesion_label = lesion_label
        
        # self.transform_post = transform_post
        #self.unlabeled_list = unlabeled_list
        self.transform_strong = transform_strong
        self.transform_weak = transform_weak
        self.crop_fn = crop_fn
        self.as_contiguous = as_contiguous
        self.data = data
        #self.labeled_ratio = labeled_ratio
        #if self.labeled_ratio is not None:
        #    split_idx = int(len(data) * labeled_ratio/100)
        #    self.labeled_list = random.sample(range(len(data)+1), split_idx)
        #    print('Chosen Labeled idx: {}'.format(self.labeled_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # train strong for batched pics
        item_strong = item
        for _transform in self.transform_strong.transforms:  # type:ignore
            # execute all the deterministic transforms
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item_strong = apply_transform(_xform, item_strong)
        
        # train weak for batched pics
        item_weak = item
        for _transform in self.transform_weak.transforms:  # type:ignore
            # execute all the deterministic transforms
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item_weak = apply_transform(_xform, item_weak)

        #if idx in self.unlabeled_list:
        #    item_strong[0]['label'] = ['None']
        #    item_strong[1]['label'] = ['None']
        #    item_weak[0]['label'] = ['None']
        #    item_weak[1]['label'] = ['None']

        item = []
        [item.extend(b) for b in [item_strong,item_weak]]
        # train weak and strong for batched pics
        #for _item in item:
        #    for _transform in self.transform_strong.transforms:  # type:ignore
        #        _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
        #        _item = apply_transform(_xform, _item)
        #        a = 1
            
        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)

        return item
    
    def collate_fn_dict(self,batches):
        
        batch = []
        [batch.extend(b) for b in batches]
        imgs = [s['image'] for s in batch]
        label = [s['label'] for s in batch]
        return {'image': imgs, 'label': label}
        

def infer_post_transforms(args, test_transforms, out_classes):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        # AsDiscreted(keys="pred", argmax=True)
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms
