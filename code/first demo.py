from torch.utils.data import Dataset
import os
from PIL import Image

                                                                                #本例前提：存放数据的文件夹名称即为label，数据为图片
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):                                    #数据集初始化函数
        self.root_dir = root_dir
        self.label_dir = label_dir                                              #输入根地址、文件夹相对地址（label）
        self.path = os.path.join(self.root_dir, self.label_dir)                 #合并即为完整的文件夹地址
        self.img_path = os.listdir(self.path)                                   #img_path为一个存放数据名称的数组

    def __getitem__(self, idx):
        img_name = self.img_path[idx]                                           #获取数据名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)   #合并出完整的数据地址（文件夹地址+数据名）
        img = Image.open(img_item_path)                                         #获取数据
        label = self.label_dir                                                  #获取文件夹相对地址（label）
        return img, label                                                       #返回数据、label

    def __len__(self):
        return len(self.img_path)                                               #返回数据集大小（数据数量）


# root_dir = "train"
# ants_label_dir = "ants_image"
# bees_label_dir = "bees_image"
# ants_dataset = MyData(root_dir, ants_label_dir)
# bees_dataset = MyData(root_dir, bees_label_dir)
# train_dataset = ants_dataset + bees_dataset                                     #合并两个数据集，前后拼接
# img, label = bees_dataset.__getitem__(0)                                       #获取特定数据集中的数据


class watermark_dataset(Dataset):
    def __init__(self, wmi, wm):
        self.wm = wm
        self.idx = list()
        for item in wmi:
            now_item = item
            self.idx.append(now_item)
        pass

    def __getitem__(self, index):
        wmi = self.idx[index]
        wm = self.wm[index]
        return wmi, wm

    def __len__(self):
        return len(self.idx)
