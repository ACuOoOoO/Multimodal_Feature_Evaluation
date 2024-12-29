
import os
import shutil
from PIL import Image
import cv2
import scipy.io as scio
import numpy as np



if not os.path.exists('VIS_IR_new_trans/test'):
    os.mkdir('VIS_IR_new_trans/test')

else:
    shutil.rmtree('VIS_IR_new_trans/test')
    os.mkdir('VIS_IR_new_trans/test')
    
if not os.path.exists('VIS_IR_new_trans/train'):
    os.mkdir('VIS_IR_new_trans/train')

else:
    shutil.rmtree('VIS_IR_new_trans/train')
    os.mkdir('VIS_IR_new_trans/train')
    
os.mkdir('VIS_IR_new_trans/test/VIS')
os.mkdir('VIS_IR_new_trans/test/IR')
os.mkdir('VIS_IR_new_trans/train/VIS')
os.mkdir('VIS_IR_new_trans/train/IR')

shutil.unpack_archive('VIS_IR_new_trans/transforms.zip', 'VIS_IR_new_trans')
shutil.unpack_archive('VIS_IR_new_trans/landmarks.zip', 'VIS_IR_new_trans/test')
test_list = open('VIS_IR_new_trans/test.txt','r').read().split('\n')
train_list = open('VIS_IR_new_trans/train.txt','r').read().split('\n')


for i,test_img in enumerate(test_list):
    if len(test_img)>2:
        if test_img.startswith('lwir'):
            try:
                vis_img = Image.open('VIS_IR_new_trans/lghd_icip2015_rgb_lwir/rgb/'+test_img.replace('png','bmp').replace('lwir','rgb'))
                # IR_img = Image.open('VIS_IR_new_trans/lghd_icip2015_rgb_lwir/lwir/'+test_img)
                # IR_img = np.array(IR_img)
                # IR_img = Image.fromarray((IR_img*255).astype(np.uint8))
                IR_img = cv2.imread('VIS_IR_new_trans/lghd_icip2015_rgb_lwir/lwir/'+test_img)
                IR_img = np.array(IR_img)
                IR_img = Image.fromarray((IR_img).astype(np.uint8))
            except:
                continue
        elif test_img.startswith('FLIR'):
            vis_img = Image.open('VIS_IR_new_trans/RoadScene/crop_LR_visible/'+test_img)
            IR_img = Image.open('VIS_IR_new_trans/RoadScene/cropinfrared/'+test_img)    
            
        if os.path.exists('VIS_IR_new_trans/test/transforms/{}.21.mat'.format(i+1)):
            H = scio.loadmat('VIS_IR_new_trans/test/transforms/{}.21.mat'.format(i+1))['H']
            IR_img = np.array(IR_img)
            IR_img = cv2.warpPerspective(IR_img,H,[IR_img.shape[1],IR_img.shape[0]])
            IR_img = Image.fromarray(IR_img)
        if os.path.exists('VIS_IR_new_trans/test/transforms/{}.12.mat'.format(i+1)):
            H = scio.loadmat('VIS_IR_new_trans/test/transforms/{}.12.mat'.format(i+1))['H']
            vis_img = np.array(vis_img)
            vis_img = cv2.warpPerspective(vis_img,H,[vis_img.shape[1],vis_img.shape[0]])
            vis_img = Image.fromarray(vis_img)
        IR_img.save('VIS_IR_new_trans/test/IR/{}.png'.format(i+1))
        vis_img.save('VIS_IR_new_trans/test/VIS/{}.png'.format(i+1))
        
for i,train_img in enumerate(train_list):
    if len(train_img)>2:
        if train_img.startswith('lwir'):
            try:
                vis_img = Image.open('VIS_IR_new_trans/lghd_icip2015_rgb_lwir/rgb/'+train_img.replace('png','bmp').replace('lwir','rgb'))
                #IR_img = Image.open('VIS_IR_new_trans/lghd_icip2015_rgb_lwir/lwir/'+train_img)
                IR_img = cv2.imread('VIS_IR_new_trans/lghd_icip2015_rgb_lwir/lwir/'+train_img)
                IR_img = np.array(IR_img)
                IR_img = Image.fromarray((IR_img).astype(np.uint8))
            except:
                continue
        elif train_img.startswith('FLIR'):
            vis_img = Image.open('VIS_IR_new_trans/RoadScene/crop_LR_visible/'+train_img)
            IR_img = Image.open('VIS_IR_new_trans/RoadScene/cropinfrared/'+train_img)    
        vis_img.save('VIS_IR_new_trans/train/VIS/{}.png'.format(i+1))
        IR_img.save('VIS_IR_new_trans/train/IR/{}.png'.format(i+1))
