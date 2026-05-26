import h5py
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors
from torch.utils.data import Dataset


PIXEL_SCALE = 80.0
THRESHOLDS = [20, 30, 35, 40]

COLOR_MAP = np.array(
    [
        [0, 0, 0, 0],
        [0, 236, 236, 255],
        [1, 160, 246, 255],
        [1, 0, 246, 255],
        [0, 239, 0, 255],
        [0, 200, 0, 255],
        [0, 144, 0, 255],
        [255, 255, 0, 255],
        [231, 192, 0, 255],
        [255, 144, 2, 255],
        [255, 0, 0, 255],
        [166, 0, 0, 255],
        [101, 0, 0, 255],
        [255, 0, 255, 255],
        [153, 85, 201, 255],
        [255, 255, 255, 255],
    ]
) / 255

BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, PIXEL_SCALE]


class ZWMOC(Dataset):
    def __init__(self, data_path, img_size, type="train"):
        super().__init__()
        assert type in ["train", "test", "val", "valid"]
        self.data_path = data_path
        self.img_size = img_size
        self.type = "test" if type in ["val", "valid"] else type
        with h5py.File(self.data_path, "r") as h5_file:
            self.all_len = int(h5_file[self.type]["all_len"][()])

    def __len__(self):
        return self.all_len

    def __getitem__(self, index):
        with h5py.File(self.data_path, "r") as h5_file:
            frames = h5_file[self.type][str(index)][()]

        frames = torch.from_numpy(frames).float() / 255.0
        if frames.ndim == 3:
            frames = frames.unsqueeze(1)
        if frames.shape[-2:] != (self.img_size, self.img_size):
            frames = F.interpolate(
                frames,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return frames


def gray2color(image, **kwargs):
    cmap = colors.ListedColormap(COLOR_MAP)
    norm = colors.BoundaryNorm(BOUNDS, cmap.N)
    return cmap(norm(image))
