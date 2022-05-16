
import os
import shutil
from PIL import Image
import cv2
import scipy.io as scio
import numpy as np



if not os.path.exists('VIS_SAR/test'):
    os.mkdir('VIS_SAR/test')

else:
    shutil.rmtree('VIS_SAR/test')
    os.mkdir('VIS_SAR/test')
    
if not os.path.exists('VIS_SAR/train'):
    os.mkdir('VIS_SAR/train')

else:
    shutil.rmtree('VIS_SAR/train')
    os.mkdir('VIS_SAR/train')
    
os.mkdir('VIS_SAR/test/VIS')
os.mkdir('VIS_SAR/test/SAR')
os.mkdir('VIS_SAR/train/VIS')
os.mkdir('VIS_SAR/train/SAR')

shutil.unpack_archive('VIS_SAR/OSdataset.zip', 'VIS_SAR')
shutil.unpack_archive('VIS_SAR/transforms.zip', 'VIS_SAR')
train_list = os.listdir('VIS_SAR/OSdataset/512/train')
test_list = os.listdir('VIS_SAR/OSdataset/512/test')
for i,train_img in enumerate(train_list):
    if len(train_img)>2:
        if train_img.startswith('opt'):
            id = train_img.replace('opt','')
            opt = 'VIS_SAR/OSdataset/512/train/opt' + id
            sar = 'VIS_SAR/OSdataset/512/train/sar' + id
            shutil.copy(opt,'VIS_SAR/train/VIS/'+id)
            shutil.copy(sar,'VIS_SAR/train/SAR/'+id)

for i,test_img in enumerate(test_list):
    if len(test_img)>2:
        if test_img.startswith('opt'):
            id = test_img.replace('opt','')
            vis_img = Image.open('VIS_SAR/OSdataset/512/test/opt' + id)
            SAR_img = Image.open('VIS_SAR/OSdataset/512/test/sar' + id)    
            
            if os.path.exists('VIS_SAR/test/transforms/{}.21.mat'.format(id.replace('.png',''))):
                H = scio.loadmat('VIS_SAR/test/transforms/{}.21.mat'.format(id.replace('.png','')))['H']
                SAR_img = np.array(SAR_img)
                SAR_img = cv2.warpPerspective(SAR_img,H,[SAR_img.shape[1],SAR_img.shape[0]])
                SAR_img = Image.fromarray(SAR_img)
            if os.path.exists('VIS_SAR/test/transforms/{}.12.mat'.format(id.replace('.png',''))):
                H = scio.loadmat('VIS_SAR/test/transforms/{}.12.mat'.format(id.replace('.png','')))['H']
                vis_img = np.array(vis_img)
                vis_img = cv2.warpPerspective(vis_img,H,[vis_img.shape[1],vis_img.shape[0]])
                vis_img = Image.fromarray(vis_img)
            SAR_img.save('VIS_SAR/test/SAR/{}.png'.format(id.replace('.png','')))
            vis_img.save('VIS_SAR/test/VIS/{}.png'.format(id.replace('.png','')))