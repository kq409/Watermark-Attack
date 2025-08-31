import numpy as np
import nn_UD_1
from Dataset import watermark_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import cv2
from DCTDWTSVD import watermark, cv_imwrite

toPIL = transforms.ToPILImage()
image_transform = transforms.Compose([transforms.ToTensor()])

UD = nn_UD_1.model_UD_1(3)
UD = UD.cuda()
checkpoint_1 = torch.load(r'E:\NN\trans\checkpoints\train_total\UD_1\KLD\finetune\UD_1_finetune_best.pth')
checkpoint_2 = torch.load(r'E:\NN\trans\checkpoints\train_total\dataset1\method1\KLD\k_fold\UD_kfold_best.pth')
checkpoint_3 = torch.load(r'E:\NN\trans\checkpoints\train_total\UD_1\MSE\finetune\UD_1_finetune_best.pth')

total_test_loss = 0

loss_mse = nn.MSELoss()
loss_mse = loss_mse.cuda()

# test_data = watermark_dataset(np.load(r'E:\NN\images\test.npy'))
# test_loader = DataLoader(test_data, batch_size=10, shuffle=True)
orig_domain = r'E:\NN\show\Iw'

UD.load_state_dict(checkpoint_3['net'])
UD.eval()

# with torch.no_grad():
#     for batch_idx, (wmi) in enumerate(test_loader):
#         wmi = wmi.cuda()
#         outputs = UD(wmi)
#         loss_2 = loss_mse(outputs, wmi)
#         total_test_loss += loss_2.item()
# print('Loss={}'.format(total_test_loss))
#
# total_test_loss = 0

for file in os.listdir(orig_domain):
    orig_name = file
    orig_path = os.path.join(orig_domain, orig_name)
    path = os.path.split(orig_path)[1]
    name = os.path.splitext(path)[0]
    wmi_tensor = image_transform(Image.open(orig_path))
    wmi_tensor = torch.reshape(wmi_tensor, (1, 3, 512, 512))
    with torch.no_grad():
        wmi_tensor = wmi_tensor.cuda()
        wm_extracted = UD(wmi_tensor)
        wm_extracted = torch.clamp(wm_extracted, min=0.0, max=1.0)
        img = toPIL(wm_extracted[0])
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
        wmimg = bwm1.extract_watermark(img_array, 1)
        cv_imwrite(r'E:\NN\show\Wa\{}.png'.format(name), wmimg)
        cv_imwrite(r'E:\NN\show\Ia\{}.png'.format(name), img_array)