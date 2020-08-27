import os
import os.path

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision import datasets
from torch.utils.data import Dataset
import cv2
import PIL

from time import time


IMG_SHAPE = (768, 768, 3)
DATASET_ROOT_FIR = "/home/esla/research/datasets/siim/chris_deotte/full_768x768_siim_kaggle"
IMAGE_FOLDER = "//home/esla/research/datasets/siim/chris_deotte/full_768x768_siim_kaggle/train"
NPY_FOLDER = "/home/esla/research/datasets/siim/chris_deotte/full_768x768_siim_kaggle/npys"
LOG_FOLDER = "logs"
CSV_LABELS_FILE = "train_siim_ohe.csv"

class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file)

        print("num of samples before cleaning: ", len(self.data))

        # esla added
        # clean out rows with all zeros (unlabelled samples)
        self.data['sum'] = self.data.sum(axis=1)

        self.data = self.data[self.data['sum'] == 1.0]
        self.data =  self.data.drop('sum', axis=1, inplace=False)
        print("num of samples after cleaning: ", len(self.data))

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)

        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        # esla debug
        print(self.class_to_idx)
        self.classes = classes

        print('\nFound {} images from {} classes.'.format(len(self.data), len(classes)))

        for class_name, idx in self.class_to_idx.items():
            # esla for debug
            print("class_name: {}, class_index: {}".format(class_name, idx))
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(class_name, idx, n_images[class_name]))
            print("\nesla debug ends")

    def __getitem__(self, index):
        path = os.path.join(self.root, self.data.loc[index, self.image_field])

        if self.add_extension:
            path = path + self.add_extension

        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[index, self.target_field]]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name


class FolderDatasetWithImgPath(datasets.ImageFolder):
    """
    Extends torchvision.datasets.ImageFolder to return
    (sample, target, path)
    """

    # Override the __getitem__ method
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        sample_target = super(FolderDatasetWithImgPath, self).__getitem__(index)
        image_path = self.imgs[index][0]
        # TBD: temporarily return image  name instead of the path for
        # compatibility. To be updated later to returning path
        image_name = os.path.basename(image_path).split('.')[0]
        #print(image_name)
        return sample_target, image_name


class FolderDatasetWithImgPathAlbum(datasets.ImageFolder):
    """
    Extends torchvision.datasets.ImageFolder to return
    (sample, target, path)
    """

    # Override the __getitem__ method
    def __getitem__no_album(self, index):
        # this is what ImageFolder normally returns
        sample_target = super(FolderDatasetWithImgPath, self).__getitem__(index)
        image_path = self.imgs[index][0]
        # TBD: temporarily return image  name instead of the path for
        # compatibility. To be updated later to returning path
        image_name = os.path.basename(image_path)
        print("sample_target type", type(sample_target[0]))
        return sample_target, image_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        img = PIL.Image.open(path)

        if self.transform is not None:
            #sample = self.transform(sample)['image']
            sample = self.transform(**{"image": np.array(img)})["image"]
            # np_img = np.asarray(sample)
            # print(type(np_img))
            # sample = self.transform(np_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target), os.path.basename(path).split('.')[0]


class DatasetFromCSV(Dataset):
    """ Do normal training
    """

    def __init__(self, data_df, soft_labels_filename=None, transforms=None):
        self.data = data
        self.transforms = transforms
        if soft_labels_filename == "":
            print("soft_labels is None")
            self.soft_labels = None
        else:
            self.soft_labels = pd.read_csv(soft_labels_filename)

    def __getitem__(self, index):
        # Read image
        # solution-1: read from raw image
        img_url = os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0] + ".jpg")
        image = cv2.cvtColor(
            cv2.imread(img_url), cv2.COLOR_BGR2RGB
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
        return (image, label), os.path.basename(img_url).split('.')[0]

    def __len__(self):
        return len(self.data)
