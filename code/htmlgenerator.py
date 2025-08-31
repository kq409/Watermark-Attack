import os

import numpy as np
from PIL import Image
import numpy

import model_eval
import Compare
import measure
# import BER
import cv2
import BER_calculate

def generate_table():
 f = open('result_table.html', 'a+')
 f.write(
  '<table border="1"><tr><th></th><th>MSE Loss</th><th>KLDiv Loss</th></tr><tr><th>Separate</th><td>SSIM={} PSNR={}</td><td>SSIM={} PSNR={}</td></tr><tr><th>Together</th><td>SSIM={} PSNR={}</td><td>SSIM={} PSNR={}</td></tr></table>'.format(
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/finetune/UD_finetune.pth', 'ssim'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/finetune/UD_finetune.pth', 'psnr'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/finetune/UD_finetune.pth', 'ssim'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/finetune/UD_finetune.pth', 'psnr'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold/UD_kfold_best.pth', 'ssim'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/MSE/k_fold/UD_kfold_best.pth', 'psnr'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/k_fold/UD_kfold_best.pth', 'ssim'),
  model_eval.compare('/home/dell/NN/checkpoints/train_total/dataset1/method1/KLD/k_fold/UD_kfold_best.pth', 'ssim')))
 f.write('</br>')

def generate_table_test():
 f = open('result_table_test.html', 'a+')
 ssim , ssim_sd, psnr, psnr_sd = model_eval.compare('/home/dell/NN/checkpoints/train_total/UD_1/KLD/finetune/UD_1_finetune_best.pth', 'UD_1')

 f.write(
  'KLD<table border="1"><tr><td>SSIM={} SSIM sd={} PSNR={} PSNR sd={}</td></tr></table>'.format(
   ssim, ssim_sd, psnr, psnr_sd
  )
 )
 f.write('</br>')

def generate_example():
 f = open('example.html', 'a+')
 #4040 4042 4059 4070 4073 4159 4205 4307 4381 4399
 for file in os.listdir('/home/dell/NN/example'):
  orig_name = file
  orig_path = os.path.join('/home/dell/NN/example', orig_name)
  path = os.path.split(orig_path)[1]
  name = os.path.splitext(path)[0]
  f.write('<img border="0" src="example/{}" height="256" width="256"><img border="0" src="example_output/image/finetune_MSE/{}_MSE_finetune.png" height="256" width="256"><img border="0" src="example_output/image/finetune_KLD/{}_KLD_finetune.png" height="256" width="256">MSE SSIM={} PSNR={} KLD SSIM={} PSNR={}<br>'.format(orig_name, name, name, Compare.compute_img('example/{}'.format(orig_name), 'example_output/image/finetune_MSE/{}_MSE_finetune.png'.format(name), 'SSIM'), Compare.compute_img('example/{}'.format(orig_name), 'example_output/image/finetune_MSE/{}_MSE_finetune.png'.format(name), 'PSNR'), Compare.compute_img('example/{}'.format(orig_name), 'example_output/image/finetune_KLD/{}_KLD_finetune.png'.format(name), 'SSIM'), Compare.compute_img('example/{}'.format(orig_name), 'example_output/image/finetune_KLD/{}_KLD_finetune.png'.format(name), 'PSNR')))
  f.write(
   '<img border="0" src="example_output/watermark/orig_extracted/{}_extracted.png" height="256" width="256"><img border="0" src="example_output/watermark/finetune_MSE/{}_MSE_finetune_extracted.png" height="256" width="256"><img border="0" src="example_output/watermark/finetune_KLD/{}_KLD_finetune_extracted.png" height="256" width="256"><img border="0" src="example_output/watermark/orig/{}" height="256" width="256"><br>'.format(
    name, name, name, orig_name))

def generate_example_UD_1():
 f = open('example_UD_1_KLD.html', 'a+')
 for file in os.listdir('/home/dell/NN/example'):
  orig_name = file
  orig_path = os.path.join('/home/dell/NN/example', orig_name)
  path = os.path.split(orig_path)[1]
  name = os.path.splitext(path)[0]
  f.write('<img border="0" src="example/{}" height="256" width="256"><img border="0" src="example_output/image/finetune_KLD/{}.png" height="256" width="256">SSIM={} PSNR={}<br>'.format(orig_name, name, Compare.compute_img('example/{}'.format(orig_name), 'example_output/image/finetune_KLD/{}.png'.format(name), 'SSIM'), Compare.compute_img('example/{}'.format(orig_name), 'example_output/image/finetune_KLD/{}.png'.format(name), 'PSNR')))
  f.write(
   '<img border="0" src="example_output/watermark/orig_extracted/{}.png" height="256" width="256"><img border="0" src="example_output/watermark/finetune_KLD/{}.png" height="256" width="256">NC={} BER={}<br>'.format(
    name, name, measure.normalizedcorrelation(numpy.array(Image.open('example_output/watermark/orig_extracted/{}.png'.format(name))), numpy.array(Image.open('example_output/watermark/finetune_KLD/{}.png'.format(name))))
    , BER_calculate.calculate(numpy.array(Image.open('example_output/watermark/orig_extracted/{}.png'.format(name))), numpy.array(Image.open('example_output/watermark/finetune_KLD/{}.png'.format(name))))))

def generate_table_wmattacked():
 f = open('result_table_attacked.html', 'a+')
 NC_sum = 0
 BER_sum = 0
 NC_list = []
 BER_list = []
 for file in os.listdir('/home/dell/NN/images/method1/test_KLDattacked_extracted'):
  orig_name = file
  orig_path = os.path.join('/home/dell/NN/images/method1/test_KLDattacked_extracted', orig_name)
  path = os.path.split(orig_path)[1]
  name = os.path.splitext(path)[0]
  NC = measure.normalizedcorrelation(numpy.array(Image.open(orig_path)), numpy.array(Image.open(os.path.join('/home/dell/NN/images/method1/test_wm', orig_name))))
  NC_sum += NC
  print(NC_sum)
  NC_list.append(NC)


  imga = Image.open(orig_path)
  imgb = Image.open(os.path.join('/home/dell/NN/images/method1/test_wm', orig_name))
  imga_array = numpy.array(imga)
  imgb_array = numpy.array(imgb)
  BER = BER_calculate.calculate(imga_array, imgb_array)
  BER_sum += BER
  print(BER_sum)
  BER_list.append(BER)

  # print(BER_sum)

 NC_sd = np.std(NC_list)
 BER_sd = np.std(BER_list)
 f.write(
  'KLD<table border="1"><tr><td>NC={} NC sd={} BER={} BER sd={}</td></tr></table>'.format(
  NC_sum/1000, NC_sd, BER_sum/1000, BER_sd),
  )
 f.write('</br>')

# if __name__ == '__main__':
#  generate()
# generate_table_test()
# generate_example()
generate_table_wmattacked()
# generate_example_UD_1()