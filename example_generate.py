import numpy as np
import nn_UD
import nn_UD_1
from Dataset import watermark_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2
from DCTDWTSVD import watermark, cv_imwrite
import os

from torch.utils.tensorboard import SummaryWriter

def example_generate():
    toPIL = transforms.ToPILImage()
    image_transform = transforms.Compose([transforms.ToTensor()])

    orig_domain = '/home/dell/NN/example'

    checkpoint_1 = torch.load(r'/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/finetune/UD_finetune.pth')
    checkpoint_2 = torch.load(r'/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/finetune/UD_finetune.pth')
    checkpoint_3 = torch.load(r'/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold/UD_kfold_best.pth')
    checkpoint_4 = torch.load(r'/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/k_fold/UD_kfold_best.pth')

    UD = nn_UD.model_UD(3)
    UD = UD.cuda()

    UD.load_state_dict(checkpoint_1['net'])
    UD.eval()

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
            wm_extracted = torch.clamp(wm_extracted, min=0.0, max=255.0)
            img = toPIL(wm_extracted[0])
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
            wmimg = bwm1.extract_watermark(img_array, 1)
            cv_imwrite('/home/dell/NN/example_output/watermark/finetune_MSE/{}_MSE_finetune_extracted.png'.format(name), wmimg)
            cv_imwrite('/home/dell/NN/example_output/image/finetune_MSE/{}_MSE_finetune.png'.format(name), img_array)

    UD.load_state_dict(checkpoint_2['net'])
    UD.eval()

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
            wm_extracted = torch.clamp(wm_extracted, min=0.0, max=255.0)
            img = toPIL(wm_extracted[0])
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
            wmimg = bwm1.extract_watermark(img_array, 1)
            cv_imwrite('/home/dell/NN/example_output/watermark/finetune_KLD/{}_KLD_finetune_extracted.png'.format(name), wmimg)
            cv_imwrite('/home/dell/NN/example_output/image/finetune_KLD/{}_KLD_finetune.png'.format(name),
                       img_array)

    UD.load_state_dict(checkpoint_3['net'])
    UD.eval()

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
            wm_extracted = torch.clamp(wm_extracted, min=0.0, max=255.0)
            img = toPIL(wm_extracted[0])
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
            wmimg = bwm1.extract_watermark(img_array, 1)
            cv_imwrite('/home/dell/NN/example_output/watermark/kfold_MSE/{}_MSE_kfold_extracted.png'.format(name), wmimg)
            cv_imwrite('/home/dell/NN/example_output/image/kfold_MSE/{}_MSE_kfold.png'.format(name), img_array)

    UD.load_state_dict(checkpoint_4['net'])
    UD.eval()

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
            wm_extracted = torch.clamp(wm_extracted, min=0.0, max=255.0)
            img = toPIL(wm_extracted[0])
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
            wmimg = bwm1.extract_watermark(img_array, 1)
            cv_imwrite('/home/dell/NN/example_output/watermark/kfold_KLD/{}_KLD_kfold_extracted.png'.format(name), wmimg)
            cv_imwrite('/home/dell/NN/example_output/image/kfold_KLD/{}_KLD_kfold.png'.format(name), img_array)

def extract(orig_domain):
    for file in os.listdir(orig_domain):
        orig_name = file
        orig_path = os.path.join(orig_domain, orig_name)
        path = os.path.split(orig_path)[1]
        name = os.path.splitext(path)[0]
        img = Image.open(orig_path)
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
        wmimg = bwm1.extract_watermark(img_array, 1)
        cv_imwrite('/home/dell/NN/images/method1/test_extracted/{}.png'.format(name),
                   wmimg)
        # print(wmimg)
        # wm_tensor = bwm1.extract_watermark(img_array, 0)
        # print(wm_tensor)

def example_generate_test():
    toPIL = transforms.ToPILImage()
    image_transform = transforms.Compose([transforms.ToTensor()])

    orig_domain = r'E:\NN\test'

    checkpoint_1 = torch.load(r'E:\NN\trans\checkpoints\train_total\UD_1\MSE\finetune\UD_1_finetune_best.pth')

    # UD = nn_UD.model_UD(3)
    UD = nn_UD_1.model_UD_1(3)
    UD = UD.cuda()

    UD.load_state_dict(checkpoint_1['net'])
    UD.eval()

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
            # cv_imwrite(r'E:\NN\example_output\watermark\finetune_MSE\{}.png'.format(name), wmimg)
            cv_imwrite(r'E:\NN\example_output\image\finetune_MSE\{}.png'.format(name), img_array)

def attack():
    toPIL = transforms.ToPILImage()
    image_transform = transforms.Compose([transforms.ToTensor()])

    orig_domain = '/home/dell/NN/images/method1/test'

    checkpoint_1 = torch.load(r'/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_best.pth')

    # UD = nn_UD.model_UD(3)
    UD = nn_UD_1.model_UD_1(3)
    UD = UD.cuda()

    UD.load_state_dict(checkpoint_1['net'])
    UD.eval()

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
            cv_imwrite('/home/dell/NN/images/method1/test_KLDattacked_extracted/{}.png'.format(name), wmimg)
            # cv_imwrite('/home/dell/NN/images/method1/test_KLDattacked/{}.png'.format(name), img_array)

# extract('/home/dell/NN/images/method1/test_attacked')
example_generate_test()
# attack()