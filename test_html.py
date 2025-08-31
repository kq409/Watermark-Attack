import os
import AvgCompare, DifCompare, CosinCompare, SSIMCompare

orig_domain = 'E:\MRI data\Test\TestImage'
#re_domain = 'E:\MRI data\Test\ReconstructionImage'
re_domain = 'E:\MRI data\Test\simg'

AS = {}
DS = {}
CS = {}
SS = {}
name_list={}


def generate():
 i = 0
 A = 0
 D = 0
 C = 0
 S = 0
 for file in os.listdir(r'E:\MRI data\Test\TestImage'):
  orig_name = file
  name_list[i] = file
  #orig_name = '1 no.jpeg'
  orig_path = os.path.join(orig_domain, orig_name)
  path = os.path.split(orig_path)[1]
  name = os.path.splitext(path)[0]
  re_name = name + '.png'
  re_path = os.path.join(re_domain, re_name)

  orig_img = '<img src="' + orig_path + '"' + 'width="300">'
  res_img = '<img src="' + re_path + '"' + 'width="300">'

  Ascore = AvgCompare.compute(orig_path, re_path)
  Dscore = DifCompare.compute(orig_path, re_path)
  Cscore = CosinCompare.compute(orig_path, re_path)
  Sscore = SSIMCompare.compute(orig_path, re_path)
  AS[i]=Ascore
  DS[i]=Dscore
  CS[i]=Cscore
  SS[i]=Sscore

  f = open('result.html', 'a+')
  f.write('Image:{}'.format(name_list[i]))
  f.write(
   '<br><img border="0" src="E:\MRI data\Test\TestImage\{}" alt="name"height="300"><img border="0" src="E:\MRI data\Test\ReconstructionImage\{}" alt="name"height="300">'.format(
    orig_name, re_name))
  f.write(
   '<table border="1"><tr><th>AverageHashScore</th><th>DifferenceHashScore</th><th>CosinScore</th><th>SSIMScore</th></tr><tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr></table>'.format(
    Ascore, Dscore, Cscore, Sscore))
  f.write('</br>')
  i = i+1
  print(i)
  f.close()
  A += Ascore
  D += Dscore
  C += Cscore
  S += Sscore
  #break
 f = open('result.html', 'a+')
 f.write('AverageHashScore={} DifferenceHashScore={} CosinScore={} SSIMScore={}'.format(A/i, D/i, C/i, S/i))


def save_html(orig_path, re_path, data_vision,model_type,image_name,cv,epoch,metric_iou,metric_dsc,metric_acc,metric_tn,metric_fp,
              metric_fn,metric_tp,metric_sensitivity,metric_specificity,processing,AUC_value):
 f = open('result.html', 'a+')
 #f.write('Image:{}'.format(name_list[i]))
 f.write('<br>< img border="0" src="./pred_finalmask_{}_{}/{}_{}_cv_{}_eopch{}.jpg" alt="name" width="300" height="300" />'.format(data_vision,model_type,processing,str(image_name)[2:-7],cv,epoch))
 f.write('<table border="1"><tr><td>IOU</td><td>{}</td><td>DSC</td><td>{}</td><td>ACC</td><td>{}</td><td>AUC</td><td>{}</td></tr><tr><td>TN</td><td>{}</td><td>FP</td><td>{}</td><td>FN</td><td>{}</td><td>TP</td><td>{}</td></tr><tr><td>Sensitivity</td><td>{}</td><td>Specificity</td><td>{}</td></tr></table>'.format(round(metric_iou,4),round(metric_dsc,4),round(metric_acc,4),round(AUC_value,4),metric_tn,metric_fp,metric_fn,metric_tp,round(metric_sensitivity,4),round(metric_specificity,4)))

 f.close()

if __name__ == '__main__':
 generate()