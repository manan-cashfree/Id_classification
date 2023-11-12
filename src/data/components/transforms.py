import random
from typing import Dict

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
import torch

train_transforms = v2.Compose([
    v2.Resize(size=(518, 518), interpolation=InterpolationMode.BICUBIC),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250]))
])

val_transforms = v2.Compose([
    v2.Resize(size=(518, 518), interpolation=InterpolationMode.BICUBIC),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250]))
])


class CustomImageFolder(ImageFolder):
    def __init__(self, transform_prob=0.3, min_crop_percent=0.70, max_crop_percent=0.75, *args, **kwargs):
        super(CustomImageFolder, self).__init__(*args, **kwargs)
        self.transform_prob = transform_prob
        self.min_crop_percent = min_crop_percent
        self.max_crop_percent = max_crop_percent
        self.transform = train_transforms

    def apply_custom_transform(self, img, target):
        if target in [self.class_to_idx['Invalid'], self.class_to_idx['PAN']]:
            return img, target
        if random.random() < self.transform_prob:  # whether to transform
            width, height = img.size  # PIL Image, so flipped
            # Apply transforms
            crop_percent = random.uniform(self.min_crop_percent, self.max_crop_percent)
            random_custom_transform = v2.RandomChoice(
                [
                    v2.RandomHorizontalFlip(p=1.0),
                    v2.RandomVerticalFlip(p=1.0),
                    v2.RandomGrayscale(p=1.0),
                    v2.RandomCrop((round(height*crop_percent), round(width*crop_percent)))
                ], p=[1, 1, 1, 9]
            )
            img = random_custom_transform(img)
            target = self.class_to_idx['Invalid']
        return img, target

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        # Apply custom transform
        img, target = self.apply_custom_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
