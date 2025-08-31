import cv2 as cv
import os
import numpy as np
import SSIMCompare
import BER_calculate

def read_path(file_pathname, array):
    i = 0
    filenames = os.listdir(file_pathname)
    filenames.sort(key=lambda x: int(x[9:-4]))
    for filename in filenames:
       # print(filename)
       orig_path = file_pathname+"/"+filename
       img = cv.imread(file_pathname+"/"+filename) #img为原图
       gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #gray原图转成灰度
       area, imgBinary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)  #imgBinary为黑白水印
       # name = os.path.splitext(filename)[0]
       path = os.path.split(orig_path)[1]
       name = os.path.splitext(path)[0]
       dest_name = name[9:] + '.png'
       des_path = os.path.join(r'/home/dell/NN/grey_to_binary', dest_name)
       path_pair = [orig_path, des_path]
       array.append(path_pair)
       # array[0][i] = orig_path
       # array[1][i] = des_path
       cv.imwrite(des_path, imgBinary)
       # print(des_path)
       # print(array[0][i])
       i += 1

# greytobinary_array =[[]]
# binary_compare_array = [[]]
# read_path(r'/home/dell/NN/download/grey_dataset/grey_dataset', greytobinary_array)
# # print(greytobinary_array)
#
#
#
#
# filenames = os.listdir(r'/home/dell/NN/download/watermark_data/watermark_data')
# filenames.sort(key=lambda x: int(x[:-4]))
# trans_filenames = os.listdir(r'/home/dell/NN/grey_to_binary')
# trans_filenames.sort(key=lambda x: int(x[:-4]))
# flag = 0
# count = 0
# miss = 0
# error = 0
# for filename in filenames:
#     for trans_filename in trans_filenames:
#         # print(trans_filename)
#         SSIM = SSIMCompare.compute_img(os.path.join(r'/home/dell/NN/download/watermark_data/watermark_data', filename), os.path.join(r'/home/dell/NN/grey_to_binary', trans_filename))
#         if SSIM>0.9:
#             if flag == 0:
#                 path_pair = [os.path.join(r'/home/dell/NN/download/watermark_data/watermark_data', filename),os.path.join(r'/home/dell/NN/grey_to_binary', trans_filename)]
#                 binary_compare_array.append(path_pair)
#             flag += 1
#             print('SSIM=',SSIM)
#     count+=1
#     if flag == 0:
#         miss+=1
#     if flag > 1:
#         error+=1
#
#     print('flag=',flag)
#     print(path_pair[1])
#     print(count/5000)
#     print('miss=', miss)
#     print('error=', error)
#     flag = 0
# binary_array = np.array(binary_compare_array)
# np.save('array.npy',binary_array)

array_read = np.load('array.npy', allow_pickle=True)
print(array_read)

# for i in range(5000):
#     i=i+1
#     img = cv.imread(greytobinary_array[i][0])
#     cv.imwrite(binary_compare_array[i][1],img)
#     print(binary_compare_array[i][1])

# for i in range(5000):
#     i=i+1
#     img = cv.imread(greytobinary_array[i][0])
#     cv.imwrite(binary_compare_array[i][1],img)
#     print(binary_compare_array[i][1])


print(1)



