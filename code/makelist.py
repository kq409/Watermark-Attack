import os
import numpy as np
from PIL import Image
from torchvision import transforms

# wmi_path = os.listdir(r'E:\NN\images\wmi')
# wmi_path = os.listdir(r'E:\NN\images\mini')
train_wmi_path = os.listdir(r'/home/dell/NN/images/method1/train')
# wm_path = os.listdir(r'E:\NN\images\wm')
test_wmi_path = os.listdir(r'/home/dell/NN/images/method1/test')
train_wmi = list()
test_wmi = list()
img_name = list()

for name in train_wmi_path:
    wmi_img_name = name
    wmi_img_path = os.path.join(r'/home/dell/NN/images/method1/train', wmi_img_name)
    train_wmi.append(wmi_img_path)
np.save('./dataset_list/method1/train_wmi.npy', train_wmi)

for name in test_wmi_path:
    wmi_img_name = name
    wmi_img_path = os.path.join(r'/home/dell/NN/images/method1/test', wmi_img_name)
    test_wmi.append(wmi_img_path)
np.save('./dataset_list/method1/test_wmi.npy', test_wmi)

# print(np.load('wmi.npy'))
# print(np.load('wmi.npy').shape)

# print('--------------------------------')
# for name in wm_path:
#     wm_img_name = name
#     wm_img_path = os.path.join(r'E:\NN\images\test\test_binary', wm_img_name)
#     wm.append(wm_img_path)
#     np.save('wm.npy', wm)
# print(np.load('wm.npy'))
# print(np.load('wm.npy').shape)

image_transform = transforms.Compose([transforms.ToTensor()])

print(np.load('./dataset_list/method1/test_wmi.npy')[0])
wmi_img = Image.open(np.load('./dataset_list/method1/test_wmi.npy')[0])
wmi_img.show()
wmi_tensor = image_transform(wmi_img)
print(np.load('./dataset_list/method1/test_wmi.npy').shape)
# print(wmi_tensor)

# print('--------------------------------')
# for name in wmi_path:
#     wmi_img_name = name
#     wmi_img_path = os.path.join(r'E:\NN\images\mini', wmi_img_name)
#     img_name.append(os.path.splitext(name)[0])
#     np.save('name.npy', img_name)
# print(np.load('name.npy'))
# print(np.load('name.npy').dtype)
# print(np.load('wmi.npy').dtype)
# print(np.load('wm.npy').dtype)