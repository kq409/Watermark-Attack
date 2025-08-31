import k_fold_check, k_fold_pretrain, k_fold_copy, finetune, k_fold_finetune

# print("开始预训练")
# k_fold_pretrain.k_fold(4, learning_rate = 0.00001, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                        epoch=50,batch_size=40,
#                        log_path = '/home/dell/NN/logs/train_total/UD_1/method1/pretrain', network='2')
#
#
# print("开始检查:预训练")
# k_fold_check.check_pretrain()
# print("开始微调:KLD")
# finetune.finetune(learning_rate=0.0002, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                   checkpoint_load_path=r'/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold_pretrain/UD_kfold_best.pth',
#                   epoch=50, loss1_factor=1, loss2_factor=1, loss_type='KLD', batch_size=40,
#                   log_path = '/home/dell/NN/logs/train_total/dataset1/method1/KLD/finetune')
# print("开始微调:MSE")
# finetune.finetune(learning_rate=0.0002, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                   checkpoint_load_path=r'/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold_pretrain/UD_kfold_best.pth',
#                   epoch=50, loss1_factor=0.000001, loss2_factor=1, loss_type='MSE', batch_size=40,
#                   log_path = '/home/dell/NN/logs/train_total/dataset1/method1/MSE/finetune')
# print("开始k折训练:MSE")
# k_fold_copy.k_fold(4, learning_rate = 0.0002, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                    epoch=50,loss1_factor=0.00001,loss2_factor=1, loss_type='MSE', batch_size=40,
#                    log_path = '/home/dell/NN/logs/train_total/dataset1/method1/MSE/k_fold')
#
# print("开始检查:MSE")
# k_fold_check.check_MSE(0.00001, 1)
#
# print("开始k折训练:KLD")
# k_fold_copy.k_fold(4, learning_rate = 0.0002, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                    epoch=50,loss1_factor=0.0000001,loss2_factor=1, loss_type='KLD', batch_size=40,
#                    log_path = '/home/dell/NN/logs/train_total/dataset1/method1/KLD/k_fold')
#
#
#
# print("开始检查:KLD")
# k_fold_check.check_KLD(0.0000001, 1)

# print("开始k_fold微调：KLD")
# k_fold_finetune.finetune(k=4,learning_rate=0.0002, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                   checkpoint_load_path=r'/home/dell/NN/checkpoints/train_total/UD_1/UD_1_kfold_best.pth',
#                   epoch=30, loss1_factor=1, loss2_factor=1, loss_type='KLD', batch_size=30,
#                   log_path = '/home/dell/NN/logs/train_total/UD_1/method1/KLD/finetune', network= '2')
print("开始检查:KLD")
k_fold_check.check_KLD(1, 1)

# print("开始k_fold微调：MSE")
# k_fold_finetune.finetune(k=4,learning_rate=0.0002, dataset_path=r'/home/dell/NN/dataset_list/method1/train_wmi.npy',
#                   checkpoint_load_path=r'/home/dell/NN/checkpoints/train_total/UD_1/UD_1_kfold_best.pth',
#                   epoch=30, loss1_factor=0.000001, loss2_factor=1, loss_type='MSE', batch_size=30,
#                   log_path = '/home/dell/NN/logs/train_total/UD_1/method1/MSE/finetune', network='2')
# print("开始检查:MSE")
# k_fold_check.check_MSE(0.000001, 1)