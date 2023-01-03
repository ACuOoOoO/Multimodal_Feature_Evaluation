import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import scipy.io as scio

def checkboard(im1,im2,d=150):
    im1 = im1*1.0
    im2 = im2*1.0
    mask = np.zeros_like(im1)
    for i in range(mask.shape[0]//d+1):
        for j in range(mask.shape[1]//d+1):
            if (i+j)%2==0:
                mask[i*d:(i+1)*d,j*d:(j+1)*d,:]+=1
    return im1*mask+im2*(1-mask)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--feature_name", type=str, default='ReDFeat',help='Name of feature')
parser.add_argument("--subsets", type=str, default='VIS_SAR',help='Type of modal: VIS_NIR, VIS_IR, VIS_SAR, '+' for all')
parser.add_argument("--nums_kp", type=int, default=-1, help="Number of feature for evluation")
args = parser.parse_args() 


import argparse
bf = cv2.BFMatcher(crossCheck=True)

lm_counter = 0
MIN_MATCH_COUNT = 5
num_black_list = 0
if args.subsets == '+': 
    subsets = ['VIS_IR','VIS_NIR','VIS_SAR']
else:
    subsets = [args.subsets]

if args.nums_kp < 0:
    nums_kp = [1024,2048,4096]
else:
    nums_kp = [args.nums_kp]
feature_name = args.feature_name

for subset in subsets:
    subset_path = os.path.join(SCRIPT_DIR,subset)
    dirlist = os.listdir(subset_path)
    if 'test' in dirlist:
        imgs = os.listdir(os.path.join(subset_path,'test','VIS'))
    else:
        continue
    print(subset)
    filepath1 = os.path.join(subset_path,'test',subset.split('_')[0])
    filepath2 = os.path.join(subset_path,'test',subset.split('_')[1])
    
    #progress_bar = tqdm(range(len(image_list)))
    for num in [1024,2048,4096]:
        errs = []
        failed_id=[]
        image_list = sorted(os.listdir(filepath1))
        img_list_whitelist = []
        progress_bar = tqdm(range(len(image_list)))
        for id in progress_bar:
            # i=id+1
            # if image_list[id] in blacklist:
            #     continue
            imgpath1 = os.path.join(filepath1, image_list[id])
            imgpath2 = os.path.join(filepath2, image_list[id])
            image1 = np.array(Image.open(imgpath1).convert('RGB'))
            image2 = np.array(Image.open(imgpath2).convert('RGB'))
            ff = image_list[id].replace('.png','.features.mat')
            feats = scio.loadmat(os.path.join(SCRIPT_DIR,'features',subset,feature_name)+'/'+ff)
            desc1 = np.array(feats['desc1'],dtype=np.float32)[0:num]
            desc2 = np.array(feats['desc2'],dtype=np.float32)[0:num]
            kp1 = np.array(feats['kp1'][:,0:2],dtype=np.float32)[0:num]
            kp2 = np.array(feats['kp2'][:,0:2],dtype=np.float32)[0:num]
            if os.path.exists(os.path.join(subset_path,'test','landmarks',image_list[id].replace('.png','.lms.mat'))):
                landmarks = scio.loadmat(os.path.join(subset_path,'test','landmarks',image_list[id].replace('.png','.lms.mat')))
                vis_lm = np.array(landmarks['vis_points'])
                ir_lm = np.array(landmarks['ir_points'])
                if len(ir_lm)<5:
                    num_black_list += 1
                    continue
            else:
                vis_lm = None
                ir_lm = None
            img_list_whitelist.append(image_list[id])
            try:
                suffix = '.12'
                H = scio.loadmat(os.path.join(subset_path,'test','transforms')+'/'+image_list[id].replace('.png',suffix+'.mat'))['H']

            except:
                suffix = '.21'
                H = scio.loadmat(os.path.join(subset_path,'test','transforms')+'/'+image_list[id].replace('.png',suffix+'.mat'))['H']

            if suffix == '.21':
                try:
                    matches = bf.match(desc1,desc2)
                except:
                    continue
                good = matches
                # matches = bf.knnMatch(desc1,desc2,k=2)
                # good = []
                # for m,n in matches:
                #     if m.distance < rt*n.distance:
                #         good.append(m)
                src_pts = np.float32([ kp1[m.queryIdx] for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx] for m in good ]).reshape(-1,1,2)
                src_im = image2
                gt_im = image1
                if not vis_lm is None:
                    lm_gt = cv2.perspectiveTransform(ir_lm.reshape(-1,1,2),H)
                    lm_src = vis_lm.reshape(-1,1,2)
            else:
                try:
                    matches = bf.match(desc2,desc1)
                except:
                    continue
                good = matches
                src_pts = np.float32([ kp2[m.queryIdx] for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp1[m.trainIdx] for m in good ]).reshape(-1,1,2)
                src_im = image1
                gt_im = image2
                if not vis_lm is None:
                    lm_gt = cv2.perspectiveTransform(vis_lm.reshape(-1,1,2),H)
                    lm_src = ir_lm.reshape(-1,1,2)

            err=10
            if len(good)>4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=10.0, maxIters=100000)
                warpim = cv2.warpPerspective(gt_im,M,[gt_im.shape[1],gt_im.shape[0]])
                im_cb = checkboard(warpim,src_im)
                im_cb = Image.fromarray(im_cb.astype(np.uint8))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results')):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results'))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset)):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name)):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset,feature_name))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'reproj')):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'reproj'))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'reproj',str(num))):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'reproj',str(num)))
                im_cb.save(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'reproj',str(num))+'/'+image_list[id].replace('_rgb.tiff','.png'))

                if M is not None:
                    err = np.linalg.norm(H-M)
                    if not vis_lm is None:
                        lm_reproj = cv2.perspectiveTransform(lm_src.reshape(-1,1,2),M)
                        err = np.sqrt(((lm_reproj-lm_gt)**2).reshape(-1,2).sum(axis=-1).mean())

                else:
                    err = 1000
            if err>=10:
                failed_id.append(image_list[id])
            errs.append(err)
        errs = np.array(errs)
        mask = errs>=10
        print('#failures is {}'.format((mask*1.0).sum()))
        if (mask*1.0).sum()==mask.shape[0]:
            print('fail to reproject')
        else:
            print('reprojection error is {}\n'.format(errs[np.logical_not(mask)].mean()))
        log_file_path = os.path.join(SCRIPT_DIR,'results',subset,feature_name,'reproj_log.txt')
        scio.savemat(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'proj_result_{}.mat'.format(num)),{'imgs':img_list_whitelist,'RE':errs})
        log_file = open(log_file_path, 'a+')
        log_file.write('#successfully repojected image: {}\n'.format(len(image_list)-(mask*1.0).sum()-num_black_list))
        log_file.write('reprojection error is {} with {} points\n'.format(errs[np.logical_not(mask)].mean(), num))
        log_file.write('failed ids: {}\n'.format(failed_id))
        log_file.close()
