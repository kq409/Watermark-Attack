import numpy as np
import nn_UD
from Dataset import watermark_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import Compare
from Compare import compute_array_ssim, compute_array_psnr
from torch.utils.tensorboard import SummaryWriter

def compare():




log_path = 'logs/logs_factors'
writer = SummaryWriter(log_path)

UD = nn_UD.model_UD(3)
UD = UD.cuda()

checkpoint_1 = torch.load(r'E:\NN\UD_kfold_3fold.pth')

batch_size = 10

test_data = watermark_dataset(np.load(r'E:\NN\images\test.npy'))
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# wmi_tensor = image_transform(Image.open(r'E:\NN\methods\watermark_dataset\test\image_cover\image_embedded4184.png'))

# wmi_tensor = torch.reshape(wmi_tensor, (1, 3, 512, 512))

# wmi_tensor = wmi_tensor.cuda()
total_test_loss = 0
UD.load_state_dict(checkpoint_1['net'])
UD.eval()
num = 0
ssim = 0
psnr = 0
ssim_sum = 0
psnr_sum = 0
with torch.no_grad():
    for batch_idx, (wmi) in enumerate(test_loader):
        wmi = wmi.cuda()
        outputs = UD(wmi)
        size = outputs.shape[0]
        for i in range(size):
            outputs_tensor = torch.reshape(outputs[i], (512, 512, 3))
            outputs_tensor = outputs_tensor.cpu()
            output_array = outputs_tensor.numpy()
            wmi_tensor = torch.reshape(wmi[i], (512, 512, 3))
            wmi_tensor = wmi_tensor.cpu()
            wmi_array = wmi_tensor.numpy()
            ssim = compute_array_ssim(output_array, wmi_array)
            psnr = compute_array_psnr(output_array, wmi_array)
            print(ssim, psnr)
            ssim_sum += ssim
            psnr_sum += psnr
            num += 1


ssim = ssim_sum / num
psnr = psnr_sum / num
print("结果：")
print(ssim, psnr)

