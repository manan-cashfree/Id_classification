from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import torch


train_transforms = v2.Compose([
    v2.Resize(size=518, interpolation=InterpolationMode.BICUBIC),
    v2.CenterCrop(size=(518, 518)),
    v2.ToTensor(),
    v2.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250]))
])

val_transforms = v2.Compose([
    v2.Resize(size=518, interpolation=InterpolationMode.BICUBIC),
    v2.CenterCrop(size=(518, 518)),
    v2.ToTensor(),
    v2.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250]))
])
