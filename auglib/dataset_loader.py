import os
import os.path

import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision import datasets


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
        image_name = os.path.basename(image_path)
        return sample_target, image_name
