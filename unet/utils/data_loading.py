import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import json
import random
import os
import os.path

# import cv2

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class BasicDataset_thumos(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')

class Thumos14(BasicDataset_thumos):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')
# def video_to_tensor(pic):
#     """Convert a ``numpy.ndarray`` to tensor.
#     Converts a numpy.ndarray (T x H x W x C)
#     to a torch.FloatTensor of shape (C x T x H x W)
    
#     Args:
#          pic (numpy.ndarray): Video to be converted to tensor.
#     Returns:
#          Tensor: Converted video.
#     """
#     return torch.from_numpy(pic.transpose([3,0,1,2]))


# def load_rgb_frames(image_dir, vid, start, num):
#   frames = []
#   for i in range(start, start+num):
#     img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
#     w,h,c = img.shape
#     if w < 226 or h < 226:
#         d = 226.-min(w,h)
#         sc = 1+d/min(w,h)
#         img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
#     img = (img/255.)*2 - 1
#     frames.append(img)
#   return np.asarray(frames, dtype=np.float32)

# def load_flow_frames(image_dir, vid, start, num):
#   frames = []
#   for i in range(start, start+num):
#     imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
#     imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
#     w,h = imgx.shape
#     if w < 224 or h < 224:
#         d = 224.-min(w,h)
#         sc = 1+d/min(w,h)
#         imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
#         imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
#     imgx = (imgx/255.)*2 - 1
#     imgy = (imgy/255.)*2 - 1
#     img = np.asarray([imgx, imgy]).transpose([1,2,0])
#     frames.append(img)
#   return np.asarray(frames, dtype=np.float32)


# def make_dataset(split_file, split, root, mode, num_classes=157):
#     dataset = []
#     with open(split_file, 'r') as f:
#         data = json.load(f)

#     i = 0
#     for vid in data.keys():
#         if data[vid]['subset'] != split:
#             continue

#         if not os.path.exists(os.path.join(root, vid)):
#             continue
#         num_frames = len(os.listdir(os.path.join(root, vid)))
#         if mode == 'flow':
#             num_frames = num_frames//2
            
#         if num_frames < 66:
#             continue

#         label = np.zeros((num_classes,num_frames), np.float32)

#         fps = num_frames/data[vid]['duration']
#         for ann in data[vid]['actions']:
#             for fr in range(0,num_frames,1):
#                 if fr/fps > ann[1] and fr/fps < ann[2]:
#                     label[ann[0], fr] = 1 # binary classification
#         dataset.append((vid, label, data[vid]['duration'], num_frames))
#         i += 1
    
#     return dataset


# class Thumos14(data_utl.Dataset):

#     def __init__(self, split_file, split, root, mode, transforms=None):
        
#         self.data = make_dataset(split_file, split, root, mode)
#         self.split_file = split_file
#         self.transforms = transforms
#         self.mode = mode
#         self.root = root

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         vid, label, dur, nf = self.data[index]
#         start_f = random.randint(1,nf-65)

#         if self.mode == 'rgb':
#             imgs = load_rgb_frames(self.root, vid, start_f, 64)
#         else:
#             imgs = load_flow_frames(self.root, vid, start_f, 64)
#         label = label[:, start_f:start_f+64]

#         imgs = self.transforms(imgs)

#         return video_to_tensor(imgs), torch.from_numpy(label)

#     def __len__(self):
#         return len(self.data)

