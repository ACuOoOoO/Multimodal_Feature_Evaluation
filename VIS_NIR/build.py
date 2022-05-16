
import os
import shutil
from PIL import Image
import cv2
import scipy.io as scio
import numpy as np
if not os.path.exists('VIS_NIR/test'):
    os.mkdir('VIS_NIR/test')

else:
    shutil.rmtree('VIS_NIR/test')
    os.mkdir('VIS_NIR/test')
    
if not os.path.exists('VIS_NIR/train'):
    os.mkdir('VIS_NIR/train')

else:
    shutil.rmtree('VIS_NIR/train')
    os.mkdir('VIS_NIR/train')
    
os.mkdir('VIS_NIR/test/VIS')
os.mkdir('VIS_NIR/test/NIR')
os.mkdir('VIS_NIR/train/VIS')
os.mkdir('VIS_NIR/train/NIR')

os.mkdir('VIS_NIR/train/NIR')
if os.path.exists('VIS_NIR/nirscene1.zip'):
    shutil.unpack_archive('VIS_NIR/nirscene1.zip', 'VIS_NIR')
else:
    raise Exception('nirscene1.zip does not exist at VIS_NIR')
    assert False,'nirscene1.zip does not exist at VIS_NIR'
shutil.unpack_archive('VIS_NIR/transforms.zip', 'VIS_NIR')
test_list = open('VIS_NIR/test.txt','r').read().split('\n')
h = 0
w = 0
train_list = open('VIS_NIR/train.txt','r').read().split('\n')
for i,train_img in enumerate(train_list):
    if len(train_img)>2:
        vis_img = Image.open(train_img.replace('data_raw','VIS_NIR')+'_rgb.tiff')
        vis_img.save('VIS_NIR/train/VIS/{}.png'.format(i+1))
        nir_img = Image.open(train_img.replace('data_raw','VIS_NIR')+'_nir.tiff')
        nir_img.save('VIS_NIR/train/NIR/{}.png'.format(i+1))

for i,test_img in enumerate(test_list):
    if len(test_img)>2:
        vis_img = Image.open(test_img.replace('data_raw','VIS_NIR')+'_rgb.tiff')
        nir_img = Image.open(test_img.replace('data_raw','VIS_NIR')+'_nir.tiff')
        if os.path.exists('VIS_NIR/test/transforms/{}.21.mat'.format(i+1)):
            H = scio.loadmat('VIS_NIR/test/transforms/{}.21.mat'.format(i+1))['H']
            nir_img = np.array(nir_img)
            nir_img = cv2.warpPerspective(nir_img,H,[nir_img.shape[1],nir_img.shape[0]])
            nir_img = Image.fromarray(nir_img)
        if os.path.exists('VIS_NIR/test/transforms/{}.12.mat'.format(i+1)):
            H = scio.loadmat('VIS_NIR/test/transforms/{}.12.mat'.format(i+1))['H']
            vis_img = np.array(vis_img)
            vis_img = cv2.warpPerspective(vis_img,H,[vis_img.shape[1],vis_img.shape[0]])
            vis_img = Image.fromarray(vis_img)
        nir_img.save('VIS_NIR/test/NIR/{}.png'.format(i+1))
        vis_img.save('VIS_NIR/test/VIS/{}.png'.format(i+1))