import os

import cv2

import numpy as np

from torch.utils.data import Dataset


class IMD2020Dataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        transforms=None
    ):

        self.image_dir = image_dir

        self.mask_dir = mask_dir

        self.transforms = transforms

        self.images = sorted(os.listdir(image_dir))


    def __len__(self):

        return len(self.images)


    def __getitem__(self,index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            image_name
        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = (mask>127).astype(np.float32)

        if self.transforms:

            aug = self.transforms(
                image=image,
                mask=mask
            )

            image = aug["image"]

            mask = aug["mask"]

        mask = mask.unsqueeze(0)

        return image,mask
