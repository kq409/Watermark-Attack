import cv2

# img_org = cv2.imread('/home/dell/NN/example/4001.png')
# img_attacked = cv2.imread('/home/dell/NN/example_output/image/finetune_MSE/4001.png')
#
# cv2.imshow('diff', img_org - img_attacked)

img_org = cv2.imread(r'E:\NN\show\Iw\image_embedded4003.png')
img_attacked = cv2.imread(r'E:\NN\show\Ia\image_embedded4003.png')

cv2.imshow('diff.jpg', img_org - img_attacked)
cv2.waitKey(0)