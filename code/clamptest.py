import torch
import numpy as np

list = np.load('./images/wmi.npy')
idx = slice(0, 1000)
list_1 = list[idx]
print(len(list_1))