import numpy as np
from PIL import Image


def calculate(imgA_path, imgB_path):
    imgA = Image.open(imgA_path)
    imgB = Image.open(imgB_path)
    imga_array = np.array(imgA)/255
    imgb_array = np.array(imgB)/255
    rows, cols = imga_array.shape
    # sqrAsum = 0
    # productsum = 0

    count = 0
    total = 1024
    for i in range(rows):
        for j in range(cols):
            if imga_array[i][j] != imgb_array[i][j]:
                count +=1

    return count