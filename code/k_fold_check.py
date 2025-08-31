import numpy as np
import nn_UD
import nn_UD_1
from Dataset import watermark_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from extract_methods import extract_and_save


def check_pretrain():
    # UD = nn_UD.model_UD(3)
    UD = nn_UD_1.model_UD_1(3)
    UD = UD.cuda()
    checkpoint_1 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_0fold.pth')
    checkpoint_2 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_1fold.pth')
    checkpoint_3 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_2fold.pth')
    checkpoint_4 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_3fold.pth')
    # checkpoint_1 = torch.load(
    #     '/home/dell/NN/checkpoints/UD_kfold_0fold.pth')
    # checkpoint_2 = torch.load(
    #     '/home/dell/NN/checkpoints/UD_kfold_1fold.pth')
    # checkpoint_3 = torch.load(
    #     '/home/dell/NN/checkpoints/UD_kfold_2fold.pth')
    # checkpoint_4 = torch.load(
    #     '/home/dell/NN/checkpoints/UD_kfold_3fold.pth')


    test_data = watermark_dataset(np.load('/home/dell/NN/dataset_list/method1/test_wmi.npy'))
    test_loader = DataLoader(test_data, batch_size=30, shuffle=True)

    toPIL = transforms.ToPILImage()
    image_transform = transforms.Compose([transforms.ToTensor()])

# wmi_tensor = image_transform(Image.open(r'E:\NN\methods\watermark_dataset\test\image_cover\image_embedded4184.png'))

# wmi_tensor = torch.reshape(wmi_tensor, (1, 3, 512, 512))

    loss_mse = nn.MSELoss()
    loss_mse = loss_mse.cuda()
    loss_kld = nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss_kld = loss_kld.cuda()

    total_test_loss = 0
    UD.load_state_dict(checkpoint_1['net'])
    UD.eval()
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            loss_2 = loss_mse(outputs, wmi)
            total_test_loss += loss_2.item()
    print('Loss={}'.format(total_test_loss))
    loss_min = total_test_loss
    set_checkpoint = 0
    checkpoint = checkpoint_1

    total_test_loss = 0
    UD.load_state_dict(checkpoint_2['net'])
    UD.eval()
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            loss_2 = loss_mse(outputs, wmi)
            total_test_loss += loss_2.item()
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 1
        checkpoint = checkpoint_2

    total_test_loss = 0
    UD.load_state_dict(checkpoint_3['net'])
    UD.eval()
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            loss_2 = loss_mse(outputs, wmi)
            total_test_loss += loss_2.item()
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 2
        checkpoint = checkpoint_3

    total_test_loss = 0
    UD.load_state_dict(checkpoint_4['net'])
    UD.eval()
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            loss_2 = loss_mse(outputs, wmi)
            total_test_loss += loss_2.item()
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 3
        checkpoint = checkpoint_4
    print(set_checkpoint)
    torch.save(checkpoint,
               '/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_kfold_best.pth')

def check_MSE(loss1_factor, loss2_factor):
    # UD = nn_UD.model_UD(3)
    UD = nn_UD_1.model_UD_1(3)
    UD = UD.cuda()
    checkpoint_1 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_0fold.pth')
    checkpoint_2 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_1fold.pth')
    checkpoint_3 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_2fold.pth')
    checkpoint_4 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_3fold.pth')

    test_data = watermark_dataset(np.load('/home/dell/NN/dataset_list/method1/test_wmi.npy'))
    test_loader = DataLoader(test_data, batch_size=40, shuffle=True)

    toPIL = transforms.ToPILImage()
    image_transform = transforms.Compose([transforms.ToTensor()])

# wmi_tensor = image_transform(Image.open(r'E:\NN\methods\watermark_dataset\test\image_cover\image_embedded4184.png'))

# wmi_tensor = torch.reshape(wmi_tensor, (1, 3, 512, 512))

    loss_mse = nn.MSELoss()
    loss_mse = loss_mse.cuda()

    total_test_loss = 0
    UD.load_state_dict(checkpoint_1['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=255.0)
            wmattacked = extract_and_save(1, outputs, test_step)
            noise_tensor = torch.randint(0, 2, (wmi.shape[0], 1, 32, 32)) * 255
            noise_tensor = noise_tensor.to(torch.float32)
            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            loss1_factor_num = loss1_factor
            loss2_factor_num = loss2_factor
            loss_1 = loss_mse(wmattacked, noise_tensor)
            loss_2 = loss_mse(outputs, wmi)
            loss = loss_1 * loss1_factor_num + loss_2 * loss2_factor_num
            total_test_loss += loss.item()

            test_step+=1
    print('Loss={}'.format(total_test_loss))
    loss_min = total_test_loss
    set_checkpoint = 0
    checkpoint = checkpoint_1

    total_test_loss = 0
    UD.load_state_dict(checkpoint_2['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=255.0)
            wmattacked = extract_and_save(1, outputs, test_step)
            noise_tensor = torch.randint(0, 2, (wmi.shape[0], 1, 32, 32)) * 255
            noise_tensor = noise_tensor.to(torch.float32)
            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            loss1_factor_num = loss1_factor
            loss2_factor_num = loss2_factor
            loss_1 = loss_mse(wmattacked, noise_tensor)
            loss_2 = loss_mse(outputs, wmi)
            loss = loss_1 * loss1_factor_num + loss_2 * loss2_factor_num
            total_test_loss += loss.item()

            test_step += 1
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 1
        checkpoint = checkpoint_2

    total_test_loss = 0
    UD.load_state_dict(checkpoint_3['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=255.0)
            wmattacked = extract_and_save(1, outputs, test_step)
            noise_tensor = torch.randint(0, 2, (wmi.shape[0], 1, 32, 32)) * 255
            noise_tensor = noise_tensor.to(torch.float32)
            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            loss1_factor_num = loss1_factor
            loss2_factor_num = loss2_factor
            loss_1 = loss_mse(wmattacked, noise_tensor)
            loss_2 = loss_mse(outputs, wmi)
            loss = loss_1 * loss1_factor_num + loss_2 * loss2_factor_num
            total_test_loss += loss.item()

            test_step += 1
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 2
        checkpoint = checkpoint_3

    total_test_loss = 0
    UD.load_state_dict(checkpoint_4['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=255.0)
            wmattacked = extract_and_save(1, outputs, test_step)
            noise_tensor = torch.randint(0, 2, (wmi.shape[0], 1, 32, 32)) * 255
            noise_tensor = noise_tensor.to(torch.float32)
            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            loss1_factor_num = loss1_factor
            loss2_factor_num = loss2_factor
            loss_1 = loss_mse(wmattacked, noise_tensor)
            loss_2 = loss_mse(outputs, wmi)
            loss = loss_1 * loss1_factor_num + loss_2 * loss2_factor_num
            total_test_loss += loss.item()

            test_step += 1
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 3
        checkpoint = checkpoint_4
    print(set_checkpoint)
    torch.save(checkpoint,
               '/home/dell/NN/checkpoints/train_total/UD_1/MSE/finetune/UD_1_finetune_best.pth')

def check_KLD(loss1_factor, loss2_factor):
    # UD = nn_UD.model_UD(3)
    UD = nn_UD_1.model_UD_1(3)
    UD = UD.cuda()
    checkpoint_1 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_0fold.pth')
    checkpoint_2 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_1fold.pth')
    checkpoint_3 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_2fold.pth')
    checkpoint_4 = torch.load('/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_3fold.pth')

    test_data = watermark_dataset(np.load('/home/dell/NN/dataset_list/method1/test_wmi.npy'))
    test_loader = DataLoader(test_data, batch_size=40, shuffle=True)

    toPIL = transforms.ToPILImage()
    image_transform = transforms.Compose([transforms.ToTensor()])

    # wmi_tensor = image_transform(Image.open(r'E:\NN\methods\watermark_dataset\test\image_cover\image_embedded4184.png'))

    # wmi_tensor = torch.reshape(wmi_tensor, (1, 3, 512, 512))

    loss_mse = nn.MSELoss()
    loss_mse = loss_mse.cuda()
    loss_kld = nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss_kld = loss_kld.cuda()

    total_test_loss = 0
    UD.load_state_dict(checkpoint_1['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            wmattacked = extract_and_save(1, outputs, test_step)  # 提取攻击后的水印，wmattacked是被攻击后的水印

            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            wmattacked_list = torch.tensor([])
            noise_tensor_list = torch.tensor([])
            for j in range(wmattacked.shape[0]):
                wmattacked_sum = torch.sum(wmattacked[j])
                zero_rate = (1024 * 255 - wmattacked_sum) / (1024 * 255)
                one_rate = wmattacked_sum / (1024 * 255)
                if wmattacked_sum == 1024 * 255:
                    zero_rate = 1 / 1024
                    one_rate = 1023 / 1024
                    print('Adjusted zero')
                if wmattacked_sum == 0:
                    zero_rate = 1023 / 1024
                    one_rate = 1 / 1024
                    print('Adjusted one')
                wmattacked_list = torch.cat((wmattacked_list, torch.tensor(
                    [[zero_rate, one_rate]])), 0)
                noise_tensor_list = torch.cat((noise_tensor_list, torch.tensor([[0.5, 0.5]])), 0)
            wmattacked_list = torch.log(wmattacked_list)
            noise_tensor_list = torch.log(noise_tensor_list)
            loss_1 = loss_kld(wmattacked_list, noise_tensor_list)

            loss_2 = loss_mse(outputs, wmi)  # 攻击前后图片做loss
            loss = loss_1 * loss1_factor + loss_2 * loss2_factor
            # loss = loss_1 + loss_2
            total_test_loss += loss.item()
    print('Loss={}'.format(total_test_loss))
    loss_min = total_test_loss
    set_checkpoint = 0
    checkpoint = checkpoint_1

    total_test_loss = 0
    UD.load_state_dict(checkpoint_2['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            wmattacked = extract_and_save(1, outputs, test_step)  # 提取攻击后的水印，wmattacked是被攻击后的水印

            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            wmattacked_list = torch.tensor([])
            noise_tensor_list = torch.tensor([])
            for j in range(wmattacked.shape[0]):
                wmattacked_sum = torch.sum(wmattacked[j])
                zero_rate = (1024 * 255 - wmattacked_sum) / (1024 * 255)
                one_rate = wmattacked_sum / (1024 * 255)
                if wmattacked_sum == 1024 * 255:
                    zero_rate = 1 / 1024
                    one_rate = 1023 / 1024
                    print('Adjusted zero')
                if wmattacked_sum == 0:
                    zero_rate = 1023 / 1024
                    one_rate = 1 / 1024
                    print('Adjusted one')
                wmattacked_list = torch.cat((wmattacked_list, torch.tensor(
                    [[zero_rate, one_rate]])), 0)
                noise_tensor_list = torch.cat((noise_tensor_list, torch.tensor([[0.5, 0.5]])), 0)
            wmattacked_list = torch.log(wmattacked_list)
            noise_tensor_list = torch.log(noise_tensor_list)
            loss_1 = loss_kld(wmattacked_list, noise_tensor_list)

            loss_2 = loss_mse(outputs, wmi)  # 攻击前后图片做loss
            loss = loss_1 * loss1_factor + loss_2 * loss2_factor
            # loss = loss_1 + loss_2
            total_test_loss += loss.item()
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 1
        checkpoint = checkpoint_2

    total_test_loss = 0
    UD.load_state_dict(checkpoint_3['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            wmattacked = extract_and_save(1, outputs, test_step)  # 提取攻击后的水印，wmattacked是被攻击后的水印

            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            wmattacked_list = torch.tensor([])
            noise_tensor_list = torch.tensor([])
            for j in range(wmattacked.shape[0]):
                wmattacked_sum = torch.sum(wmattacked[j])
                zero_rate = (1024 * 255 - wmattacked_sum) / (1024 * 255)
                one_rate = wmattacked_sum / (1024 * 255)
                if wmattacked_sum == 1024 * 255:
                    zero_rate = 1 / 1024
                    one_rate = 1023 / 1024
                    print('Adjusted zero')
                if wmattacked_sum == 0:
                    zero_rate = 1023 / 1024
                    one_rate = 1 / 1024
                    print('Adjusted one')
                wmattacked_list = torch.cat((wmattacked_list, torch.tensor(
                    [[zero_rate, one_rate]])), 0)
                noise_tensor_list = torch.cat((noise_tensor_list, torch.tensor([[0.5, 0.5]])), 0)
            wmattacked_list = torch.log(wmattacked_list)
            noise_tensor_list = torch.log(noise_tensor_list)
            loss_1 = loss_kld(wmattacked_list, noise_tensor_list)

            loss_2 = loss_mse(outputs, wmi)  # 攻击前后图片做loss
            loss = loss_1 * loss1_factor + loss_2 * loss2_factor
            # loss = loss_1 + loss_2
            total_test_loss += loss.item()
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 2
        checkpoint = checkpoint_3

    total_test_loss = 0
    UD.load_state_dict(checkpoint_4['net'])
    UD.eval()
    test_step = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(test_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            wmattacked = extract_and_save(1, outputs, test_step)  # 提取攻击后的水印，wmattacked是被攻击后的水印

            wmattacked = torch.tensor(wmattacked, requires_grad=True)
            outputs = torch.tensor(outputs, requires_grad=True)
            wmattacked_list = torch.tensor([])
            noise_tensor_list = torch.tensor([])
            for j in range(wmattacked.shape[0]):
                wmattacked_sum = torch.sum(wmattacked[j])
                zero_rate = (1024 * 255 - wmattacked_sum) / (1024 * 255)
                one_rate = wmattacked_sum / (1024 * 255)
                if wmattacked_sum == 1024 * 255:
                    zero_rate = 1 / 1024
                    one_rate = 1023 / 1024
                    print('Adjusted zero')
                if wmattacked_sum == 0:
                    zero_rate = 1023 / 1024
                    one_rate = 1 / 1024
                    print('Adjusted one')
                wmattacked_list = torch.cat((wmattacked_list, torch.tensor(
                    [[zero_rate, one_rate]])), 0)
                noise_tensor_list = torch.cat((noise_tensor_list, torch.tensor([[0.5, 0.5]])), 0)
            wmattacked_list = torch.log(wmattacked_list)
            noise_tensor_list = torch.log(noise_tensor_list)
            loss_1 = loss_kld(wmattacked_list, noise_tensor_list)

            loss_2 = loss_mse(outputs, wmi)  # 攻击前后图片做loss
            loss = loss_1 * loss1_factor + loss_2 * loss2_factor
            # loss = loss_1 + loss_2
            total_test_loss += loss.item()
    print('Loss={}'.format(total_test_loss))
    if total_test_loss <= loss_min:
        loss_min = total_test_loss
        set_checkpoint = 3
        checkpoint = checkpoint_4
    print(set_checkpoint)
    torch.save(checkpoint,
               '/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_best.pth')
# check()