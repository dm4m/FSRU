"""
An Lao
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset):
        # self.text = torch.from_numpy(np.array(dataset['text']))
        self.text = torch.from_numpy(dataset['text'])
        self.image = list(dataset['image'])  # torch.from_numpy(dataset['image'])
        self.label = torch.from_numpy(dataset['label'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.text[item], self.image[item], self.label[item]
