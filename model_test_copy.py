import numpy as np
import nn_UD
from Dataset import watermark_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

UD = nn_UD.model_UD(3)
UD = UD.cuda()

checkpoint_1 = torch.load(r'/home/dell/NN/checkpoints/UD_kfold_0fold.pth')
checkpoint_2 = torch.load(r'/home/dell/NN/checkpoints/UD_kfold_1fold.pth')
checkpoint_3 = torch.load(r'/home/dell/NN/checkpoints/UD_kfold_2fold.pth')
checkpoint_4 = torch.load(r'/home/dell/NN/checkpoints/UD_kfold_3fold.pth')


test_data = watermark_dataset(np.load(r'/home/dell/NN/dataset_list/test_wmi.npy'))
test_loader = DataLoader(test_data, batch_size=40, shuffle=True)

toPIL = transforms.ToPILImage()
image_transform = transforms.Compose([transforms.ToTensor()])

# wmi_tensor = image_transform(Image.open(r'E:\NN\methods\watermark_dataset\test\image_cover\image_embedded4184.png'))

# wmi_tensor = torch.reshape(wmi_tensor, (1, 3, 512, 512))

loss_mse = nn.MSELoss()
# loss_mse = nn.L1Loss()
loss_mse = loss_mse.cuda()

# wmi_tensor = wmi_tensor.cuda()
total_test_loss = 0
UD.load_state_dict(checkpoint_1['net'])
UD.eval()
with torch.no_grad():
    for batch_idx, (wmi) in enumerate(test_loader):
        wmi = wmi.cuda()
        outputs = UD(wmi)
        loss_2 = loss_mse(outputs, wmi)
        total_test_loss += loss_2.item()
print('Loss={}'.format(total_test_loss))

total_test_loss = 0
UD.load_state_dict(checkpoint_2['net'])
UD.eval()
with torch.no_grad():
    for batch_idx, (wmi) in enumerate(test_loader):
        wmi = wmi.cuda()
        outputs = UD(wmi)
        loss_2 = loss_mse(outputs, wmi)
        total_test_loss += loss_2.item()
print('Loss={}'.format(total_test_loss))

total_test_loss = 0
UD.load_state_dict(checkpoint_3['net'])
UD.eval()
with torch.no_grad():
    for batch_idx, (wmi) in enumerate(test_loader):
        wmi = wmi.cuda()
        outputs = UD(wmi)
        loss_2 = loss_mse(outputs, wmi)
        total_test_loss += loss_2.item()
print('Loss={}'.format(total_test_loss))

total_test_loss = 0
UD.load_state_dict(checkpoint_4['net'])
UD.eval()
with torch.no_grad():
    for batch_idx, (wmi) in enumerate(test_loader):
        wmi = wmi.cuda()
        outputs = UD(wmi)
        loss_2 = loss_mse(outputs, wmi)
        total_test_loss += loss_2.item()
print('Loss={}'.format(total_test_loss))