import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
from skimage.feature import match_descriptors
import scipy.io as scio
import argparse

blacklist_NIR = ['89.png', '87.png', '105.png', '129.png']

parser = argparse.ArgumentParser()
parser.add_argument("--feature_name", type=str, default='ReDFeat',help='Name of feature')
parser.add_argument("--subsets", type=str, default='VIS_SAR',help='Type of modal: VIS_NIR, VIS_IR, VIS_SAR, '+' for all')
parser.add_argument("--nums_kp", type=int, default=-1, help="Number of feature for evluation")
parser.add_argument("--vis_flag", type=bool, default=True. help="Visualization flag")
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
vis_flag = args.vis_flag
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

    for num in [1024,2048,4096]:
    # for num in [2048]:
        N_k1 = []
        N_k2 = []
        N_corr = []
        N_corretmatches = []
        N_in_corretmatches = []
        image_list = sorted(os.listdir(filepath1))
        img_list_whitelist = []
        progress_bar = tqdm(range(len(image_list)))
        for id in progress_bar:
            # i=id+1
            if subset=='NIR' and image_list[id] in blacklist_NIR:
                continue
            else:
                img_list_whitelist.append(image_list[id])
            imgpath1 = os.path.join(filepath1, image_list[id])
            imgpath2 = os.path.join(filepath2, image_list[id])
            image1 = np.array(Image.open(imgpath1).convert('RGB'))
            image2 = np.array(Image.open(imgpath2).convert('RGB'))
            ff = image_list[id].replace('.png','.features.mat')
            feats = scio.loadmat(os.path.join(SCRIPT_DIR,'features',subset,feature_name,ff))
            desc1 = feats['desc1']
            desc2 = feats['desc2']
            kp1 = feats['kp1'][:,0:2]
            kp2 = feats['kp2'][:,0:2]
            try:
                suffix = '.12'
                H = scio.loadmat(os.path.join(subset_path,'test','transforms')+'/'+image_list[id].replace('.png',suffix+'.mat'))['H']
                ones = np.ones([np.size(kp2,0),1])
                kp_ir_warped = np.hstack([kp2,ones])
                kp_ir_warped = H @ kp_ir_warped.transpose()
                kp_ir_warped = kp_ir_warped/kp_ir_warped[2,:]
                kp_ir_warped = kp_ir_warped[0:2,:].transpose()
                kp_vis_warped = kp1
            except:
                suffix = '.21'
                H = scio.loadmat(os.path.join(subset_path,'test','transforms')+'/'+image_list[id].replace('.png',suffix+'.mat'))['H']
                ones = np.ones([np.size(kp1,0),1])
                kp_vis_warped = np.hstack([kp1,ones])
                kp_vis_warped = H @ kp_vis_warped.transpose()
                kp_vis_warped = kp_vis_warped/kp_vis_warped[2,:]
                kp_vis_warped = kp_vis_warped[0:2,:].transpose()
                kp_ir_warped = kp2

            

            N_k1.append(kp1[0:num].shape[0])
            N_k2.append(kp2[0:num].shape[0])
            
            #################################
        
            keypoints1_=torch.FloatTensor(kp_vis_warped)[0:num][:,:2]
            keypoints2_=torch.FloatTensor(kp_ir_warped)[0:num][:,:2]
            x_1 = keypoints1_[:,0]
            y_1 = keypoints1_[:,1]
            x_2 = keypoints2_[:,0]
            y_2 = keypoints2_[:,1]
            x_dist = (x_1.unsqueeze(1)-x_2.unsqueeze(0)).pow(2)
            y_dist = (y_1.unsqueeze(1)-y_2.unsqueeze(0)).pow(2)
            dist_k = x_dist + y_dist
            n_corr = ((dist_k<=9).float().sum(dim=1)>0.9).float().sum()
            N_corr.append(n_corr.cpu().item())
            #################################
            

            
            # Mutual nearest neighbors matching
            # In [6]:
            matches = match_descriptors(desc1[0:num], desc2[0:num], cross_check=True)
            # In [7]:
            # print('Number of raw matches: %d.' % matches.shape[0])
            # Number of raw matches: 296.
            # Homography fitting
            # In [8]:
            keypoints_left = kp_vis_warped[0:num][matches[:, 0], : 2]
            keypoints_left_raw = kp1[0:num][matches[:, 0], : 2]
            keypoints_right = kp_ir_warped[0:num][matches[:, 1], : 2]
            keypoints_right_raw = kp2[0:num][matches[:, 1], : 2]
            
            dif = (keypoints_left - keypoints_right)
            dist_m = dif[:, 0]**2 + dif[:, 1]**2
            inds = dist_m<=9
            #print(inds.sum())
            N_corretmatches.append(inds.sum())
            
            if vis_flag:
                keypoints_right_raw_inlier = keypoints_right_raw[inds]
                keypoints_left_raw_inlier = keypoints_left_raw[inds]
                
                inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left_raw_inlier]
                inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right_raw_inlier]
                placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(inds.sum())]
                image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results')):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results'))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset)):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name)):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset,feature_name))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match')):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match'))
                if not os.path.exists(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match',str(num))):
                    os.mkdir(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match',str(num)))
                #print(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match',str(num))+'/'+image_list[id].replace('_rgb.tiff','.png'))
                cv2.imwrite(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match',str(num))+'/'+image_list[id].replace('_rgb.tiff','.png'),cv2.cvtColor(image3, cv2.COLOR_RGB2BGR))
            np.random.seed(0)
            progress_bar.set_postfix(Image_id = id)

        N_corr = np.array(N_corr)*1.0
        N_k1 = np.array(N_k1)*1.0
        N_k2 = np.array(N_k2)*1.0
        N_corretmatches = np.array(N_corretmatches)*1.0
        #N_in_corretmatches = np.array(N_in_corretmatches)
        rep_rate = N_corr*1.0/np.array([N_k1,N_k2]).min(axis=0)
        mat_rate = N_corretmatches*1.0/N_corr
        from scipy.io import savemat
        savemat(os.path.join(os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match_result_{}.mat'.format(num))),{'imgs':img_list_whitelist,'N_corr':N_corr,'N_k1':N_k1,'N_k2':N_k2,'N_correctmatches':N_corretmatches})
        print('Number of infrared keypoints: %f.' % np.mean(N_k1))
        print('Number of visible keypoints: %f.' % np.mean(N_k2))
        print('Number of correspondence: %f.' % N_corr.mean())
        print('Number of correct matches: %f.' % np.mean(N_corretmatches))
        #print('Number of inlier correct matches: %d.' % np.mean(N_in_corretmatches))
        print('repeated rate: {}.'.format(rep_rate.mean()))
        print('matched rate: {}.'.format(mat_rate.mean()))

        log_file_path = os.path.join(SCRIPT_DIR,'results',subset,feature_name,'match_log.txt')
        if os.path.exists(log_file_path):
            print('[Warning] Log file already exists.')
        log_file = open(log_file_path, 'a+')
        log_file.write('Number of infrared keypoints: %f.\n' % np.mean(N_k1))
        log_file.write('Number of visible keypoints: %f.\n' % np.mean(N_k2))
        log_file.write('Number of correspondence: %f.\n' % N_corr.mean())
        log_file.write('Number of correct matches: %f.\n' % np.mean(N_corretmatches))
        #log_file.write('Number of inlier correct matches: %d.\n' % np.mean(N_in_corretmatches))
        log_file.write('repeated rate: {}.\n'.format(rep_rate.mean()))
        log_file.write('matched rate: {}.\n'.format(mat_rate.mean()))
        log_file.close()
