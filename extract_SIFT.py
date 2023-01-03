from re import sub
import torch
import os 
import cv2
import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from extract_patches.core import extract_patches
import torch.nn as nn
import scipy.io as scio
from copy import deepcopy
import time
os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch.nn as nn
AP = nn.AvgPool2d(9,stride=1,padding=4)
MP = nn.AvgPool2d(9,stride=1,padding=4)
def get_SIFT_keypoints(sift, img, tic=False):

    # convert to gray-scale and compute SIFT keypoints
    time_ = 0
    since = time.time()
    keypoints = sift.detect(img, None)
    time_ += time.time()-since
    # keypoints = sift.detect(img, None)
    img_tensor = torch.from_numpy(img*1.0/255).permute(2,0,1).unsqueeze(0)
    mask_extra = (MP((img_tensor>1e-12).sum(dim=1,keepdim=True).float())>1e-5).float()
    for ii in range(2):
        mask_extra = (AP(mask_extra)>0.9999).float()
    mask_extra = mask_extra.squeeze(0).squeeze(0).numpy()
    response = []

    for kp in keypoints:
        x,y = kp.pt
        x = min(int(x+0.5),mask_extra.shape[0]-1)
        y = min(int(y+0.5),mask_extra.shape[1]-1)
        t = mask_extra[y,x]
        response.append(t*kp.response)
    #response = np.array([kp.response for kp in keypoints])
    since =time.time()
    respSort = np.argsort(response)[::-1]

    pt = np.array([kp.pt for kp in keypoints])[respSort]
    size = np.array([kp.size for kp in keypoints])[respSort]
    angle = np.array([kp.angle for kp in keypoints])[respSort]
    response = np.array([kp.response for kp in keypoints])[respSort]
    time_ += time.time()-since
    #print(time_)
    if tic:
        return pt, size, angle, response,time_
    return pt, size, angle, response

def get_SIFT(sift,img_dir,args):
    img = cv2.cvtColor(cv2.imread(img_dir),cv2.COLOR_BGR2RGB)
    keypoints, scales, angles, responses= get_SIFT_keypoints(sift,
                                                            img)
    kpts = [
        cv2.KeyPoint(
            x=keypoints[i][0],
            y=keypoints[i][1],
            _size=scales[i],
            _angle=angles[i]) for i in range(min(args.num_features,len(scales)))
    ]
    desc = sift.compute(img,kpts)[1]*1.0
    desc = desc/np.linalg.norm(desc,axis=1, keepdims=True)
    kp = deepcopy(keypoints[0:min(args.num_features,len(scales))])
    return desc,kp

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--subsets", type=str, default='+', help='number of keypoints')
    parser.add_argument("--num_features", type=int, default=4096, help='number of keypoints')
    parser.add_argument("--contrastThreshold", type=float, default=-10000, help='number of keypoints')
    parser.add_argument("--edgeThreshold", type=float, default=-10000, help='number of keypoints')
    args = parser.parse_args()
    if args.subsets=='+':
        args.subsets=['VIS_NIR','VIS_IR','VIS_SAR']
    else:
        args.subsets = [args.subsets]
    feature_name = 'SIFT'
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=args.contrastThreshold, edgeThreshold=args.edgeThreshold)
    if not os.path.exists(os.path.join(SCRIPT_DIR,'features')):
        os.mkdir(os.path.join(SCRIPT_DIR,'features'))
    for subset in args.subsets:
        if not os.path.exists(os.path.join(SCRIPT_DIR,'features',args.subsets[0])):
            os.mkdir(os.path.join(SCRIPT_DIR,'features',args.subsets[0]))
        if not os.path.exists(os.path.join(SCRIPT_DIR,'features',args.subsets[0],feature_name)):
            os.mkdir(os.path.join(SCRIPT_DIR,'features',args.subsets[0],feature_name))
        type1 = subset.split('_')[0]
        type2 = subset.split('_')[1]
        imgs = os.listdir(os.path.join(subset,'test',type1))
        for k,img in enumerate(imgs):
            img1_dir = os.path.join(subset,'test',type1,img)
            img2_dir = os.path.join(subset,'test',type2,img)
            desc1,kp1 = get_SIFT(sift,img1_dir,args)
            desc2,kp2 = get_SIFT(sift,img2_dir,args)
            scio.savemat(os.path.join('features',feature_name,subset,img.replace('.png','.features.mat')),
                        {'desc1':desc1,
                        'kp1':kp1,
                        'desc2':desc2,
                        'kp2':kp2})


    