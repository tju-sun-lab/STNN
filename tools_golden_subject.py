import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import os
import glob
import torchvision.transforms as transforms




class cwtDataset(torch.utils.data.Dataset):

    def __init__(self, root1, root2, root3):

        self.files_A = root1
        self.files_B = torch.from_numpy(root2)
        self.files_C = root3

    def __getitem__(self, index):
        # item_A = self.files_A[random.randint(0, len(self.files_A) - 1)]
        item_B = self.files_B[index % len(self.files_B)]
        item_A = self.files_A[index % len(self.files_A)]
        item_C = self.files_C[index % len(self.files_C)]

        return {"A": item_A, "B": item_B, "C":item_C}

    def __len__(self):

        return max(len(self.files_A), len(self.files_B))
        # return len(self.files_A)


class val_cwtDataset(torch.utils.data.Dataset):
    def __init__(self, root1):


        self.files_A = torch.from_numpy(root1)


    def __getitem__(self, index):
        item_A = self.files_A[index]
        return item_A

    def __len__(self):

        return len(self.files_A)

