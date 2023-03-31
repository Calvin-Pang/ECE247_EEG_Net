import torch
import os
import numpy as np
from torch.utils.data import Dataset
from dataset.augmentation import smooth

class EEG_Patch_Dataset(Dataset):
    def __init__(self, train_dir, label_path, transform=None):
        """
        Args:
            data (np.ndarray): A numpy array of shape (N, H, W, C), where N is the number of images, H is the height,
                W is the width, and C is the number of channels.
            targets (np.ndarray): A numpy array of shape (N,) containing the class labels for each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir = train_dir
        self.samples = os.listdir(train_dir)
        self.labels = np.load(label_path) - 769
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = int(self.samples[index].split('_')[0]) - 1
        sample =  torch.from_numpy(np.load(os.path.join(self.dir, self.samples[index]))).float().unsqueeze(0)
        label = torch.from_numpy(np.array(self.labels[sample_id])).long()

        if self.transform:
            sample = self.transform(sample)

        return sample, label