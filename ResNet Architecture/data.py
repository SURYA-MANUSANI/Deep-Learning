from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_frame = self.data.to_numpy()
        label = data_frame[index][1:]
        label = np.array(label, dtype=float)
        path = data_frame[index][0]
        imge = imread(path)
        imge = gray2rgb(imge)
        return self._transform(imge), torch.tensor(label)
