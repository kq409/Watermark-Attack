import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import nn_UD, nn_CA
from Dataset import watermark_dataset
from extract_methods import extract_and_save


def get_k_fold_data(k, i, all_train_data, all_train_data_list):
    # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
    assert k > 1
    fold_size = len(all_train_data) // k  # 每份的个数:数据总条数/折数（向下取整）

    X_train_list = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
        X_part_list = all_train_data_list[idx]
        # X_part = watermark_dataset(X_part_list)
        # X_part = X[idx]  # 只对第一维切片即可
        if j == i:  # 第i折作test
            X_valid_list = X_part_list
            print('valid')
            print(len(X_valid_list))
        elif X_train_list is None:
            X_train_list = X_part_list
            print('train')
            print(len(X_train_list))
        else:
            # X_train = torch.cat((X_train, X_part), dim=0)  # 其他剩余折进行拼接 也仅第一维
            X_train_list = np.append(X_train_list, X_part_list)
            print('train')
            print(len(X_train_list))
    return X_train_list, X_valid_list


def k_fold(k, learning_rate, dataset_path, epoch, loss1_factor, loss2_factor, loss_type, batch_size, log_path):
    train_loss_sum, test_loss_sum = 0, 0
    all_train_data = watermark_dataset(np.load(dataset_path))
    all_train_data_size = len(all_train_data)
    all_train_data_list = np.load(dataset_path)
    # epoch = 50
    # learning_rate = 0.0002
    # batch_size = 32
    is_hold = 0

    writer = SummaryWriter(log_path)

    for i in range(k):
        wmi_train_list, wmi_valid_list = get_k_fold_data(k, i, all_train_data=all_train_data, all_train_data_list=all_train_data_list)  # 获取第i折交叉验证的训练和验证数据
        wmi_train = watermark_dataset(wmi_train_list)
        wmi_valid = watermark_dataset(wmi_valid_list)
        UD = nn_UD.model_UD(3)
        UD = UD.cuda()

        optimizer_1 = torch.optim.Adam(UD.parameters(), lr=learning_rate)

        loss_mse = nn.MSELoss()
        loss_mse = loss_mse.cuda()
        loss_kld = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss_kld = loss_kld.cuda()

        total_train_step = 0
        loss_min = 0
        loss2_min = 0

        print("训练集的长度为：{}".format(len(wmi_train_list)))
        print("验证集的长度为：{}".format(len(wmi_valid_list)))
        print("----------第{}折训练开始----------".format(i + 1))

        for num in range(epoch):
            train_step = 0
            UD.train()
            print("----------第{}次训练开始----------".format(num + 1))

            checkpoint = {
                "net": UD.state_dict(),
                'optimizer': optimizer_1.state_dict(),
                "epoch": num
            }

            train_loader = DataLoader(wmi_train, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(wmi_valid, batch_size=batch_size, shuffle=True)

            for batch_idx, (wmi) in enumerate(train_loader):
                wmi = wmi.cuda()  # wmi是加了水印的图片，wm是水印GT
                outputs = UD(wmi)  # outputs是被攻击后的水印图片
                outputs = torch.clamp(outputs, min=0.0, max=255.0)

                wmattacked = extract_and_save(1, outputs, train_step)  # 提取攻击后的水印，wmattacked是被攻击后的水印

                noise_tensor = torch.randint(0, 2, (wmi.shape[0], 1, 32, 32)) * 255
                noise_tensor = noise_tensor.to(torch.float32)
                wmattacked = torch.tensor(wmattacked, requires_grad=True)
                outputs = torch.tensor(outputs, requires_grad=True)
                # print(wmattacked.shape, noise_tensor.shape)
                loss1_factor_num = loss1_factor
                loss2_factor_num = loss2_factor

                if loss_type == 'KLD':
                    noise_tensor = torch.where(noise_tensor <= 1, 0.000001, noise_tensor)
                    wmattacked = torch.where(wmattacked<=1, 0.000001, wmattacked)
                    wmattacked = torch.log(wmattacked)
                    wmattacked = torch.tensor(wmattacked, requires_grad=True)
                    noise_tensor = torch.log(noise_tensor)
                    loss_1 = loss_kld(wmattacked, noise_tensor)

                if loss_type == 'MSE':
                    loss_1 = loss_mse(wmattacked, noise_tensor)                  #被攻击后的水印与随机噪声做loss
                    # if is_hold = 1:
                    #     loss1_factor_num = 0
                    if loss_1.item() <= 6000:
                        # is_hold = 1
                        loss1_factor_num = 0
                    else:
                        loss1_factor_num = loss1_factor

                loss_2 = loss_mse(outputs, wmi)  # 攻击前后图片做loss
                loss = loss_1 * loss1_factor_num + loss_2 * loss2_factor_num    #权重调整
                # loss = loss_1 + loss_2

                writer.add_scalar("Loss", loss, total_train_step)
                writer.add_scalar("Loss_1", loss_1, total_train_step)
                writer.add_scalar("Loss_2", loss_2, total_train_step)
                # writer.add_scalar("Loss_2", loss_2, total_train_step)
                writer.add_image("output", outputs[0], 0)

                # print(wmi.shape)
                optimizer_1.zero_grad()
                # loss_2.backward()
                loss.backward()
                optimizer_1.step()
                train_step += 1
                total_train_step += 1

                if train_step % 10 == 0:
                    # print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
                    # print("第{}折,第{}epoch 训练次数：{}, Loss1:{} Loss2:{} 总Loss:{}".format(i+1, num+1, train_step,
                    #                                                                  loss_1.item(), loss_2.item(),
                    #                                                                  loss.item()))
                    print("第{}折,第{}epoch 训练次数：{}, Loss:{}, Loss1:{}, Loss2:{} ".format(i + 1, num + 1, train_step,
                                                                                             loss.item(), loss_1.item(),
                                                                                              loss_2.item(),))

            # torch.save(checkpoint, './checkpoints/UD_pretrain.pth')

            UD.eval()
            total_validation_loss = 0
            with torch.no_grad():
                for batch_idx, (wmi) in enumerate(validation_loader):
                    wmi = wmi.cuda()
                    outputs = UD(wmi)
                    wmattacked = extract_and_save(1, outputs, train_step)  # 提取攻击后的水印，wmattacked是被攻击后的水印


                    noise_tensor = torch.randint(0, 2, (wmi.shape[0], 1, 32, 32)) * 255
                    noise_tensor = noise_tensor.to(torch.float32)
                    wmattacked = torch.tensor(wmattacked, requires_grad=True)
                    outputs = torch.tensor(outputs, requires_grad=True)
                    # print(wmattacked.shape, noise_tensor.shape)
                    # loss_1 = loss_mse(wmattacked, noise_tensor)  # 被攻击后的水印与随机噪声做loss

                    if loss_type == 'KLD':
                        noise_tensor = torch.where(noise_tensor <= 1, 0.000001, noise_tensor)
                        wmattacked = torch.where(wmattacked <= 1, 0.000001, wmattacked)
                        wmattacked = torch.log(wmattacked)
                        wmattacked = torch.tensor(wmattacked, requires_grad=True)
                        noise_tensor = torch.log(noise_tensor)
                        loss_1 = loss_kld(wmattacked, noise_tensor)

                    if loss_type == 'MSE':
                        loss_1 = loss_mse(wmattacked, noise_tensor)  # 被攻击后的水印与随机噪声做loss
                        # if is_hold = 1:
                        #     loss1_factor_num = 0
                        if loss_1.item() <= 32000:
                            # is_hold = 1
                            loss1_factor_num = 0
                        else:
                            loss1_factor_num = loss1_factor

                    loss_2 = loss_mse(outputs, wmi)  # 攻击前后图片做loss
                    loss = loss_1 * loss1_factor + loss_2 * loss2_factor
                    # loss = loss_1 + loss_2
                    total_validation_loss += loss.item()

            if num == 0:
                loss_min = total_validation_loss
                if loss_type=='KLD':
                    torch.save(checkpoint,
                               '/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/k_fold/UD_kfold_{}fold.pth'.format(
                                   i))
                else:
                    torch.save(checkpoint, '/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold/UD_kfold_{}fold.pth'.format(i))
                print('Model Saved')
            if loss_min > total_validation_loss:
                loss_min = total_validation_loss
                if loss_type == 'KLD':
                    torch.save(checkpoint,
                               '/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/k_fold/UD_kfold_{}fold.pth'.format(
                                   i))
                else:
                    torch.save(checkpoint,
                               '/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold/UD_kfold_{}fold.pth'.format(
                                   i))
                print('Model Saved')
            print('Loss_min = {}'.format(loss_min))
            # print("第{}epoch测试集上的Loss:{}".format(i+1, total_validation_loss))
            print("第{}折 第{}epoch测试集上的Loss:{}".format(i + 1,num + 1, total_validation_loss))
            torch.cuda.empty_cache()
    writer.close()


# k_fold(4, learning_rate = 0.00001, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy', epoch=50,loss1_factor=0.00001,
#        loss2_factor=1, loss_type='KLD', batch_size=40, log_path = '/home/dell/NN/logs/train_total/dataset1/method1/KLD/k_fold')