import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.model import MMNet

import scipy.io as scio
from copy import deepcopy
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    
def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    model = MMNet()
    weights = checkpoint['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return model.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.6):
        super(NonMaxSuppression,self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr
        
    def forward(self, repeatability):
        #repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima*0
        border_mask[:,:,10:-10,10:-10]=1
        maxima = maxima*border_mask
        print(maxima.sum())
        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, image_type,
                        scale_f=2**0.25, min_scale=0.0, 
                        max_scale=1, min_size=256, 
                        max_size=1024, verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            with torch.no_grad():
                if image_type == '1':
                    descriptors, repeatability = net.forward1(img)
                elif image_type == '2':
                    descriptors, repeatability = net.forward2(img)

            mask = repeatability*0
            mask[:,:,args.border:-args.border,args.border:-args.border] = 1
            repeatability=repeatability*mask
            y,x = detector(repeatability) # nms
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]
            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            #S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    #S = torch.cat(S) # scale
    scores = torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--subsets", type=str, default='VIS_NIR', help='VIS_IR, VIS_NIR, VIS_SAR')
    parser.add_argument("--num_features", type=int, default=4096, help='Number of features')
    parser.add_argument("--model", type=str, default='/data1/ACuO/MMFeat-master/Pretrained/VIS_NIR.pth', help='model path')

    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--border", type=float, default=5) 
    parser.add_argument("--reliability-thr", type=float, default=0.5)
    parser.add_argument("--repeatability-thr", type=float, default=0.4)

    parser.add_argument("--gpu", type=int, default=0, help='use -1 for CPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    args = parser.parse_args()
    if args.subsets=='+':
        args.subsets=['VIS_NIR','VIS_IR','VIS_SAR']
    else:
        args.subsets = [args.subsets]
    feature_name = 'MMFeat'
    net = load_network(args.model)
    net = net.cuda()
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)
    if not os.path.exists(os.path.join(SCRIPT_DIR,'features')):
        os.mkdir(os.path.join(SCRIPT_DIR,'features'))
    if not os.path.exists(os.path.join(SCRIPT_DIR,'features',args.subsets[0])):
        os.mkdir(os.path.join(SCRIPT_DIR,'features',args.subsets[0]))
    if not os.path.exists(os.path.join(SCRIPT_DIR,'features',args.subsets[0],feature_name)):
        os.mkdir(os.path.join(SCRIPT_DIR,'features',args.subsets[0],feature_name))
    type1 = args.subsets[0].split('_')[0]
    type2 = args.subsets[0].split('_')[1]
    for subset in args.subsets:
        # load the network...
        file_path = os.path.join(args.subsets[0],'test',type1)
        if not os.path.exists(file_path):
            file_path = os.path.join(SCRIPT_DIR,args.subsets[0],'test',type1)
        imgs = os.listdir(file_path)
        imgs = sorted(imgs)
        time_ = 0
        for i,img in enumerate(imgs):
            if img.endswith('.png'):
                t = deepcopy(img)
                img = os.path.join(args.subsets[0],'test',type1,t)
                if not os.path.exists(img):
                    img = os.path.join(SCRIPT_DIR,args.subsets[0],'test',type1,t)
                img = Image.open(img).convert('RGB')
                W, H = img.size
                img = TF.to_tensor(img).unsqueeze(0)
                img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
                img = img.cuda()
                # extract keypoints/descriptors for a single image
                xys, desc, scores = extract_multiscale(net, img, detector, '1',
                    scale_f   = args.scale_f, 
                    min_scale = args.min_scale, 
                    max_scale = args.max_scale,
                    min_size  = args.min_size, 
                    max_size  = args.max_size, 
                    verbose = True)
                if len(scores)<args.num_features:
                    idxs = scores.topk(len(scores))[1]
                else:
                    idxs = scores.topk(args.num_features)[1]
                kp1 = xys[idxs].cpu().numpy()
                desc1 = desc[idxs].cpu().numpy()
                img = os.path.join(args.subsets[0],'test',type2,t)
                if not os.path.exists(img):
                    img = os.path.join(SCRIPT_DIR,args.subsets[0],'test',type2,t)
                img = Image.open(img).convert('RGB')
                W, H = img.size
                img = TF.to_tensor(img).unsqueeze(0)
                img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
                img = img.cuda()
                
                # extract keypoints/descriptors for a single image
                xys, desc, scores = extract_multiscale(net, img, detector, '2',
                    scale_f   = args.scale_f, 
                    min_scale = args.min_scale, 
                    max_scale = args.max_scale,
                    min_size  = args.min_size, 
                    max_size  = args.max_size, 
                    verbose = True)
                if len(scores)<args.num_features:
                    idxs = scores.topk(len(scores))[1]
                else:
                    idxs = scores.topk(args.num_features)[1]
                kp2 = xys[idxs].cpu().numpy()
                desc2 = desc[idxs].cpu().numpy()
                scio.savemat(os.path.join(SCRIPT_DIR,'features',args.subsets[0],feature_name,t.replace('.png','.features.mat')),
                {'desc1':desc1,
                'kp1':kp1,
                'desc2':desc2,
                'kp2':kp2})



