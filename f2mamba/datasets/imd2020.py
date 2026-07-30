import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class IMD2020Dataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_list=None,
        transforms=None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        if image_list is None:
            self.images = sorted(os.listdir(image_dir))
        else:
            self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # binary mask
        mask = (mask > 127).astype(np.float32)

        if self.transforms:
            transformed = self.transforms(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0)

        return image, mask
