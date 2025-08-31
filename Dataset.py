from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision
from torchvision import transforms
import numpy as np

image_transform = transforms.Compose([transforms.ToTensor()])

class watermark_dataset(Dataset):
    def __init__(self, wmi):                                             #转化为ndarray?
        self.idx = list()
        for item in wmi:
            now_item = item
            self.idx.append(now_item)
        pass

    def __getitem__(self, index):
        wmi = self.idx[index]
        wmi_tensor = image_transform(Image.open(wmi))

        return wmi_tensor

    def __len__(self):
        return len(self.idx)                                  #获取特定数据集中的数据