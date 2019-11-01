from __future__ import absolute_import
from __future__ import division
import argparse
from functools import partial
from config import cfg, assert_and_infer_cfg
import logging
import math
import os
import sys

import torch
import numpy as np

from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from utils.f_boundary import eval_mask_boundary
import datasets
import loss
import network
import optimizer

# Argument Parser
parser = argparse.ArgumentParser(description='GSCNN')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
parser.add_argument('--dataset', type=str, default='cityscapes')
parser.add_argument('--cv', type=int, default=0,
                    help='cross validation split')
parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,
                    help='joint loss')
parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class')
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Rescaled LR Rate')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Rescaled Poly')

parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--att_weight', type=float, default=1.0,
                    help='Attention loss weight for joint loss')
parser.add_argument('--dual_weight', type=float, default=1.0,
                    help='Dual loss weight for joint loss')

parser.add_argument('--evaluate', action='store_true', default=False)

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--sgd_finetuned',action='store_true',default=False)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=175)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--rotate', type=float,
                    default=0, help='rotation')
parser.add_argument('--gblur', action='store_true', default=True)
parser.add_argument('--bblur', action='store_true', default=False) 
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=1)
parser.add_argument('--bs_mult_val', type=int, default=2)
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Synchronized BN')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (1 epoch run ) to verify nothing failed')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

#Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1
#Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

criterion, criterion_val = loss.get_loss(args)
net = network.get_net(args, criterion)


class GscnnModel(object):
	'''load the model'''
    def __init__(self, frame,net):
      
        self.device = torch.device('cuda' if self.configer.exists('gpu_id') else 'cpu')
        self.frame = frame

        self.model =net
    	self.model.load_state_dict(torch.load('./checkpoint/best_epoch_11_mean-iu_0.80997.pth'))


    def run():

    	net.eval()

        with torch.no_grad():
            seg_out, edge_out = self.model(self.frame)

        return seg_out,  edge_out


'''
        seg_predictions = seg_out.data.max(1)[1].cpu()
        edge_predictions = edge_out.max(1)[0].cpu()
'''

def run_visualization(frame,net):
  """Inferences DeepLab model and visualizes result."""


	MODEL=GscnnModel(frame,net)
#  original_im = Image.fromarray(frame)
	seg_map = MODEL.run()

  #image = apply_mask(resized_im, seg_map)
  #seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #image =seg_image
  #cv2.imshow('video',)


	mask = np.zeros([len(seg_map),len(seg_map[0])])
	mask = mask.astype('int8')
	for i in range(len(seg_map)):
    	for j in range(len(seg_map[0])):
        	if seg_map[i][j] == 15 :

          		mask[i][j] = 255
          
	return mask  





def main():
	capture =cv2.VideoCapture(0)

	while True:

		ret,frame = capture.read()

		#get the mask from the model and visualization
		mask = run_visualization(frame,net)

		image=np.array(frame)
		dim =(640,480) ## image resize large

		#project the mask into image 
		res = cv2.biswise_and(image,image,mask=mask)
		res_resized = cv2.resize(res,dim,interpolation =cv2.INTER_AREA)

		cv2.imshow('video',res)


		 #out.write(image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



capture.release()
cv2.destroyAllWindows()






















'''
checkpoint = {'model': ClassNet(),
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epoch': epoch}
​
torch.save(checkpoint, 'checkpoint.pkl')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer = TheOptimizerClass()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    
    return model
    
model = load_checkpoint('checkpoint.pkl')

'''