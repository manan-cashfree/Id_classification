import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2 import functional as F

plt.rcParams["savefig.bbox"] = 'tight'


def plot(imgs: torch.Tensor, row_title=None, **imshow_kwargs):
    """Perform a single validation step on a batch of data from the validation set.

    :param imgs: A batch of images [N,C,H,W]
    :param row_title: Title for a row. Default None
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        print(f'single image')
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    print(f'num_row: {num_rows}, num_cols: {num_cols}')
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            img = F.to_image(img)
            img = F.to_dtype(img, torch.uint8, scale=True)
            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            print(f'type(img): {type(img)}, shape: {img.shape}')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
