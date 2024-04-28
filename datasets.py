import os, random, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class SeismicDataset(Dataset):
    def __init__(self, path, frames = 7, frame_distance = 5, augmentation_transforms=None):
        self.seismic_files = self.get_file_list(path)
        self.frames = frames
        self.frame_distance = frame_distance
        self.augmentation_transforms = augmentation_transforms

    def __len__(self):
        return len(self.seismic_files)

    def __getitem__(self, idx):
        seismic_path = self.seismic_files[idx]
        seismic_data = np.load(seismic_path)

        H, W, L = seismic_data.shape

        roi_width = self.frames * self.frame_distance
        offset = np.random.randint(H - roi_width, size=1)[0]
        frames_of_interest  = offset + np.array(range(0, roi_width, self.frame_distance))
        seismic_data = seismic_data[frames_of_interest, :, :]
        seismic_data = torch.from_numpy(seismic_data)

        seismic_data = self.augment_image(seismic_data)

        return seismic_data

    def get_file_list(self, path):
        dirs = [os.path.join(path, f) for f in os.listdir(path)]
        self.file_list = dirs

        return self.file_list

    def augment_image(self, image_t):
        if np.random.randint(2) == 1:
            image_t = v2.functional.rotate(image_t, angle =180)
    
        return image_t