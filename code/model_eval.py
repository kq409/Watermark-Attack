import numpy as np
import nn_UD, nn_UD_1
from Dataset import watermark_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import Compare



# checkpoint_1 = torch.load(r'E:\NN\checkpoints\UD_kfold_0fold.pth')
# checkpoint_2 = torch.load(r'E:\NN\checkpoints\UD_kfold_1fold.pth')
# checkpoint_3 = torch.load(r'E:\NN\checkpoints\UD_kfold_2fold.pth')
# checkpoint_4 = torch.load(r'E:\NN\checkpoints\UD_kfold_3fold.pth')




toPIL = transforms.ToPILImage()
image_transform = transforms.Compose([transforms.ToTensor()])


def compare(checkpoint_path, model_type):
    ssim_sum = 0
    psnr_sum = 0
    num = 0
    ssim_list = []
    psnr_list = []
    test_data = watermark_dataset(np.load('/home/dell/NN/dataset_list/test_wmi.npy'))
    test_loader = DataLoader(test_data, batch_size=40)
    checkpoint = torch.load(checkpoint_path)
    if model_type == 'UD':
        UD = nn_UD.model_UD(3)
    if model_type == 'UD_1':
        UD = nn_UD_1.model_UD_1(3)
    UD = UD.cuda()
    UD.load_state_dict(checkpoint['net'])
    UD.eval()
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            for i in range(outputs.shape[0]):
                outputs_tensor = torch.reshape(outputs[i], (512, 512, 3))
                outputs_tensor = outputs_tensor.cpu()
                output_array = outputs_tensor.numpy()
                wmi_tensor = torch.reshape(wmi[i], (512, 512, 3))
                wmi_tensor = wmi_tensor.cpu()
                wmi_array = wmi_tensor.numpy()
                ssim = Compare.compute_array_ssim(output_array, wmi_array)
                psnr = Compare.compute_array_psnr(output_array, wmi_array)
                print(ssim, psnr)
                ssim_list.append(ssim)
                psnr_list.append(psnr)
                ssim_sum += ssim
                psnr_sum += psnr
                ssim_sd = np.std(ssim_list)
                psnr_sd = np.std(psnr_list)
                num += 1
    print('SSIM={}, PSNR={}'.format(ssim_sum/num, psnr_sum/num))
    # if return_type == 'ssim':
    return ssim_sum/num, ssim_sd, psnr_sum/num, psnr_sd
    # if return_type == 'psnr':
    #     return

# s, p = compare('/home/dell/NN/checkpoints/UD_kfold_3fold.pth')
# print(s, p)