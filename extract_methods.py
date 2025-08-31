import torch

import DCTDWTSVD
from PIL import Image
from torchvision import transforms
import DCTDWTSVD
from DCTDWTSVD import watermark, cv_imwrite
import numpy as np
import cv2

image_transform = transforms.Compose([transforms.ToTensor()])

def extract_and_save(method, nnoutput, step):
    toPIL = transforms.ToPILImage()
    size = nnoutput.shape[0]
    if method == 1:
        bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
        for i in range(size):
            if (step + 1) % 10 == 0:                                #每10个step储存一次结果
                img = toPIL(nnoutput[i])                            #转化为PIL image
                # img_array = np.array(nnoutput[i].cpu())                   #转化为numpy数组
                # img_array = nnoutput[i]
                # img_array = img_array.detach().numpy
                # img_array = img_array[:, :, ::-1].copy()
                # img_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                ouput_save_path = './output/' + 'output' + '.jpg'
                attackedwm_save_path = "./attackedwm/" + 'wm' + ".png"
                if i == 0:
                    wm_tensor_tmp = bwm1.extract_watermark(img_array, 0)
                    wm_tensor = torch.reshape(wm_tensor_tmp, (1, 1, 32, 32))
                    img.save(ouput_save_path)
                    img = bwm1.extract_watermark(img_array, 1)
                    cv_imwrite(attackedwm_save_path, img)
                    print('Image Saved')
                else:
                    wm_tensor_tmp = bwm1.extract_watermark(img_array, 0)
                    wm_tensor_tmp = torch.reshape(wm_tensor_tmp, (1, 1, 32, 32))
                    wm_tensor = torch.cat([wm_tensor, wm_tensor_tmp], dim=0)

            else:                                                      #否则不储存
                img = toPIL(nnoutput[i])
                # img_array = nnoutput[i]
                # img_array = img_array.detach().numpy                    # 转化为numpy数组
                # img_array = img_array[:, :, ::-1].copy()
                # img_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if i == 0:
                    wm_tensor_tmp = bwm1.extract_watermark(img_array, 0)
                    wm_tensor = torch.reshape(wm_tensor_tmp, (1, 1, 32, 32))
                else:
                    wm_tensor_tmp = bwm1.extract_watermark(img_array, 0)
                    wm_tensor_tmp = torch.reshape(wm_tensor_tmp, (1, 1, 32, 32))
                    wm_tensor = torch.cat([wm_tensor, wm_tensor_tmp], dim=0)
    wm_tensor = wm_tensor.type(torch.float32)
    return wm_tensor

    # img = toPIL(nnoutput)
    # ouput_save_path = './output/' + name + '.jpg'
    # attackedwm_save_path = "./attackedwm/" + name + ".png"
    # img.save(ouput_save_path)
    # if method == 1:
    #     bwm1 = watermark(4399, 2333, 36, 20, wm_shape=(32, 32))
    #     for i in range(size):
    #         bwm1.extract(ouput_save_path, attackedwm_save_path)
    #         wm_tensor = image_transform(Image.open(attackedwm_save_path))
    # return wm_tensor
