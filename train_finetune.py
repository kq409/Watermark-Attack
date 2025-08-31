import torch
from torch import nn
import numpy as np
from PIL import Image

import nn_CA, nn_UD
import torch.utils.data as tdata
from torch.utils.data import DataLoader
from Dataset import watermark_dataset

from torch.utils.tensorboard import SummaryWriter

# import randomgenerator

from extract_methods import extract_and_save

# print(np.load(r'E:\NN\images\wmi.npy'))

batch_size = 40

loss1_factor = 0.0001
loss2_factor = 1.0

log_path = 'logs/finetune'
checkpoint_1 = torch.load(r'/home/dell/NN/checkpoints/UD_kfold_3fold.pth')


all_train_data = watermark_dataset(np.load(r'/home/dell/NN/dataset_list/train_wmi.npy'))        #预留填写训练数据地址
all_train_data_size = len(all_train_data)
train_size = 3000
validation_size = all_train_data_size - train_size
train_data, validation_data = tdata.random_split(all_train_data, [train_size, validation_size])

# test_data = watermark_dataset(np.load(), np.load())                 #预留填写测试数据地址

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train_data_size = len(train_data)
# test_data_size = len(test_data)

print("训练集的长度为：{}".format(train_data_size))
# print("测试集的长度为：{}".format(test_data_size))




# CA = nn_CA.model_CA(3)
UD = nn_UD.model_UD(3)
UD = UD.cuda()
UD.load_state_dict(checkpoint_1['net'])

learning_rate = 0.0001
total_train_step = 0
total_validation_step = 0

epoch = 50

loss_mse = nn.MSELoss()
# loss_mse = nn.L1Loss()
loss_mse = loss_mse.cuda()
loss_min = 0

optimizer_1 = torch.optim.Adam(UD.parameters(), lr=learning_rate)

writer = SummaryWriter(log_path)

i=0
for i in range(epoch):
    UD.train()
    print("----------第{}次训练开始----------".format(i+1))

    checkpoint = {
        "net": UD.state_dict(),
        'optimizer': optimizer_1.state_dict(),
        "epoch": i
    }

    for batch_idx, (wmi) in enumerate(train_loader):
        wmi = wmi.cuda()                                        #wmi是加了水印的图片，wm是水印GT
        outputs  = UD(wmi)                                      #outputs是被攻击后的水印图片
        outputs = torch.clamp(outputs, min=0.0, max=255.0)
        wmattacked = extract_and_save(1, outputs, total_train_step)                         #提取攻击后的水印，wmattacked是被攻击后的水印
        for j in range(wmi.shape[0]):
            noise_name = np.random.randint(0, 10000)
            if j == 0:
                noise = np.load('./noise/' + str(noise_name) + '.npy')                    #生成随机噪声
                noise_tensor = torch.from_numpy(noise)
                noise_tensor = torch.reshape(noise_tensor, (1, 1, 32, 32))

            else:
                noise = np.load('./noise/' + str(noise_name) + '.npy')
                noise_tensor_tmp = torch.from_numpy(noise)
                noise_tensor_tmp = torch.reshape(noise_tensor_tmp, (1, 1, 32, 32))
                noise_tensor = torch.cat((noise_tensor, noise_tensor_tmp), 0)

        noise_tensor = noise_tensor.float()
        wmattacked = torch.tensor(wmattacked, requires_grad=True)
        # outputs = torch.tensor(outputs, requires_grad=True)
        # print(wmattacked.shape, noise_tensor.shape)
        loss_1 = loss_mse(wmattacked, noise_tensor)                  #被攻击后的水印与随机噪声做loss
        loss_2 = loss_mse(outputs, wmi)                         #攻击前后图片做loss
        loss = loss_1 * loss1_factor + loss_2 * loss2_factor    #权重调整

        writer.add_scalar("Loss", loss, total_train_step)
        writer.add_scalar("Loss_1", loss_1, total_train_step)
        writer.add_scalar("Loss_2", loss_2, total_train_step)
        writer.add_image("output", outputs[0], 0)
        writer.add_image("wm_extracted", wmattacked[0], 0)

        # print(wmi.shape)
        optimizer_1.zero_grad()
        # loss_2.backward()
        loss.backward()
        optimizer_1.step()
        total_train_step+=1

        # print(1)

        if total_train_step % 10 == 0:
            # print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
    # torch.save(checkpoint, './checkpoints/UD_ckpt_{}.pth'.format(i))

    UD.eval()
    total_validation_loss = 0
    with torch.no_grad():
        for batch_idx, (wmi) in enumerate(validation_loader):
            wmi = wmi.cuda()
            outputs = UD(wmi)
            wmattacked = extract_and_save(1, outputs, total_train_step)
            for j in range(wmi.shape[0]):
                noise_name = np.random.randint(0, 10000)
                if j == 0:
                    noise = np.load('./noise/' + str(noise_name) + '.npy')  # 生成随机噪声
                    noise_tensor = torch.from_numpy(noise)
                    noise_tensor = torch.reshape(noise_tensor, (1, 1, 32, 32))

                else:
                    noise = np.load('./noise/' + str(noise_name) + '.npy')
                    noise_tensor_tmp = torch.from_numpy(noise)
                    noise_tensor_tmp = torch.reshape(noise_tensor_tmp, (1, 1, 32, 32))
                    noise_tensor = torch.cat((noise_tensor, noise_tensor_tmp), 0)

            loss_1 = loss_mse(wmattacked, noise_tensor)
            loss_2 = loss_mse(outputs, wmi)
            loss = loss_1 * loss1_factor + loss_2 * loss2_factor
            total_validation_loss += loss.item()

    writer.add_scalar("Validation Loss", total_validation_loss, i)

    # if i == 0:
    #     loss_min = total_validation_loss
    #     torch.save(checkpoint, './checkpoints/UD_train.pth')
    #     print('Model Saved')
    # if loss_min > total_validation_loss:
    #     loss_min = total_validation_loss
    #     torch.save(checkpoint, './checkpoints/UD_train.pth')
    #     print('Model Saved')

    # loss_min = total_validation_loss
    torch.save(checkpoint, './checkpoints/UD_train.pth')
    print('Model Saved')

    print("整体测试集上的Loss:{}".format(total_validation_loss))




    # torch.save(checkpoint, 'UD_ckpt_{}.pth'.format(i))
    # torch.save(UD.state_dict(), "UD_{}.pkl".format(i+1))

# torch.save(UD.state_dict(), "UD_{}.pkl".format(epoch))




                                                #由于用到BN与Dropout，记得启用valid和train
        # if i == 0:
        #     print(wmi)
        #     print(wm)
        #     print(i)
        #     # wmi_img = Image.open(wmi[i])
        #     # wm_img = Image.open(wm[i])
        #     # wmi_img.show()
        #     # wm_img.show()
        # i+=1

#Totensor会把每个像素都除以255！因此还原后要乘255