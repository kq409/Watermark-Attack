import numpy as np
import os
import DCTDWTSVD
from DCTDWTSVD import watermark

from torch.utils.tensorboard import SummaryWriter

# wmi_path = r'E:\NN\sample\image_embedded845.png'
# wmi_attacked_path = r'E:\NN\sample\testoutput.jpg'
# bwm1 = watermark(4399,2333,36,20,wm_shape=(32,32))
#
# # bwm1.extract(wmi_path, '1.png')
#
# bwm1.extract(wmi_attacked_path, './sample/testoutput_extracted.png')

list = np.load(r'E:\NN\trans\dataset_list\test_wmi.npy')
print(list.shape)

# writer = SummaryWriter('logs_test')
# for i in range(100):
#     writer.add_scalar("y=2x", 3*i, i)
#
# writer.close()