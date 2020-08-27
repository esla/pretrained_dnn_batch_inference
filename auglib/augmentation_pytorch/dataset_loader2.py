# @Author: yican, yelanlan
# @Date: 2020-05-27 22:58:45
# @Last Modified by:   yican
# @Last Modified time: 2020-05-27 22:58:45

# Standard libraries
import os
from time import time

# Third party libraries
import cv2
import numpy as np
import pandas as pd
import torch
import imutils
from torchvision.utils import save_image


from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    Resize,
    CenterCrop,
    Rotate,
    ShiftScaleRotate,
    VerticalFlip,
)
from torch.utils.data import DataLoader, Dataset

# User defined libraries
from utils import IMAGE_FOLDER, IMG_SHAPE
#from torchsampler import ImbalancedDatasetSampler

# for fast read data
# from utils import NPY_FOLDER


class DatasetFromCSV(Dataset):
    """ Do normal training
    """

    def __init__(self, data, soft_labels_filename=None, transforms=None):
        self.data = data
        self.transforms = transforms
        if soft_labels_filename == "":
            print("soft_labels is None")
            self.soft_labels = None
        else:
            self.soft_labels = pd.read_csv(soft_labels_filename)

    def __getitem__(self, index):

        start_time = time()
        # Read image
        # solution-1: read from raw image
        image = cv2.cvtColor(
            cv2.imread(os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0] + ".jpg")), cv2.COLOR_BGR2RGB
        )
        # esla temporarily added
        #image = imutils.resize(image, width=1200)

        # esla added  the following few lines below for debugging
        # self.dest_dir_images = "/home/esla/research/datasets/siim/my_new/1200_width_resized/images/"
        # self.dest_dir_npy = "/home/esla/research/datasets/siim/my_new/1200_width_resized/npys/"
        # dest_file_wo_ext_img = self.dest_dir_images + self.data.iloc[index, 0]
        # dest_file_wo_ext_npy = self.dest_dir_npy + self.data.iloc[index, 0]
        # tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if not os.path.exists(self.dest_dir_images + self.data.iloc[index, 0] + ".jpg"):
        #     print(f"saving image: {dest_file_wo_ext_img}")
        #     cv2.imwrite(dest_file_wo_ext_img+".jpg", tmp_image)
        #     # save as npy
        #     np.save(dest_file_wo_ext_npy +".npy", image)

        #print(f'image: {os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0] + ".jpg")}')
        # solution-2: read from npy file which can speed the data load time.
        # image = np.load(os.path.join(NPY_FOLDER, "raw", self.data.iloc[index, 0] + ".npy"))

        # Convert if not the right shape
        if image.shape != IMG_SHAPE:
            image = image.transpose(1, 0, 2)

        # Do data augmentation
        if self.transforms is not None:
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)


        # Soft label
        if self.soft_labels is not None:
            label = torch.FloatTensor((self.data.iloc[index, 1:].values * 0.7).astype(np.float) +
                                      (self.soft_labels.iloc[index, 1:].values * 0.3).astype(np.float))
        else:
            label = torch.FloatTensor(self.data.iloc[index, 1:].values.astype(np.int64))

        # esla debug
        #print(f'image: {os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0] + ".jpg")}')
        #print(f'label: {label}')
        return image, label, time() - start_time

    def __len__(self):
        return len(self.data)


def generate_transforms(image_size):

    train_transform = Compose(
        [
            #CenterCrop(height=1024, width=640),
            Resize(height=image_size[0], width=image_size[1]),
            OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    val_transform = Compose(
        [
            #CenterCrop(height=1024, width=640),
            Resize(height=image_size[0], width=image_size[1]),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    return {"train_transforms": train_transform, "val_transforms": val_transform}


def generate_dataloaders(hparams, train_data, val_data, transforms):
    train_dataset = DatasetFromCSV(
        data=train_data, transforms=transforms["train_transforms"], soft_labels_filename=hparams.soft_labels_filename
    )
    val_dataset = DatasetFromCSV(
        data=val_data, transforms=transforms["val_transforms"], soft_labels_filename=hparams.soft_labels_filename
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.train_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # esla note: on the callback_get_label https://github.com/ufoym/imbalanced-dataset-sampler/issues/8
    #train_dataloader = DataLoader(
    #    train_dataset,
    #    sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda x, i: tuple(x[i][1].tolist())),
    #    batch_size=hparams.train_batch_size,
    #    num_workers=hparams.num_workers,
    #    pin_memory=True,
    #    drop_last=True,
    #)


    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.val_batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        #pin_memory=True,
        pin_memory=False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader
