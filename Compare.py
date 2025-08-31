
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

#from scipy.misc import imread
import cv2
import numpy as np

def compute_img(path1, path2, type):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    if type == 'SSIM':
        SSIM = structural_similarity(img1, img2, multichannel=True)
        return SSIM
    if type == 'PSNR':
        PSNR = peak_signal_noise_ratio(img1, img2)
        return PSNR

def compute_array_ssim(img1, img2):
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    SSIM = structural_similarity(img1, img2, multichannel=True)
    return SSIM

def compute_array_psnr(img1, img2):
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    PSNR = peak_signal_noise_ratio(img1, img2)
    return PSNR
# print(1)
# print(compute_img(r'E:\NN\sample\testoutput.jpg', r'E:\NN\sample\image_embedded4184.png'))