import torch
import torch.cuda
import random
import numpy as np
import torch
import time
import torch.nn as nn
import copy
import torchvision as tv
import torchvision.models as models

import os 
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from utils import AverageMeter, accuracy
import argparse
import onnxruntime as onnxrt
import PIL

from torchvision.models import resnet18

model = resnet18(pretrained=True)
model = model.cuda()

def work_init(work_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + work_id)
    np.random.seed(seed + work_id)

def model_eval(images_dir, image_size, batch_size=64, num_workers=16, quant = False, model_name=None):
    
    data_loader_kwargs = { 'worker_init_fn':work_init, 'num_workers' : num_workers}
    val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    val_data = datasets.ImageFolder(images_dir, val_transforms)
    val_dataloader = DataLoader(val_data, batch_size, shuffle = False, pin_memory = True, **data_loader_kwargs)
    
    def func_wrapper(model, arguments):
        top1_acc = 0.0
        top5_acc = 0.0
        total_num = 0
        iterations , use_cuda = arguments[0], arguments[1]
        if use_cuda:
            model.cuda()
        top1_accuracy = AverageMeter('top1Acc')
        top5_accuracy = AverageMeter('top5Acc')
        for sample, label in tqdm(val_dataloader):
            total_num += sample.size()[0]
            if use_cuda:
                sample = sample.cuda()
                label = label.cuda()
            logits = model(sample)
            pred = torch.argmax(logits, dim = 1)
            correct = sum(torch.eq(pred, label)).cpu().numpy()
            
            
            top1,top5 = accuracy(logits, label, (1,5))
            top1_accuracy.update(top1[0])
            top5_accuracy.update(top5[0])
            
            top1_acc += correct
        avg_acc = top1_acc * 100. / total_num
        #print("Top 1 ACC : {:0.2f}".format(avg_acc))
        print("####################################")
        print("\nCalculate "+model_name+" accuracy\n")
        print('Top1 : {:0.6f}'.format(top1_accuracy.avg))
        print('Top5 : {:0.6f}'.format(top5_accuracy.avg))
        
        #return top1_accuracy
    
    return func_wrapper

from aimet_torch.model_preparer import prepare_model
prepared_model = prepare_model(model)

from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters


images_dir = "/home/rayari/5000"
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
batch_size = 64

val_data = datasets.ImageFolder(images_dir, val_transforms)
val_dataloader = DataLoader(val_data, batch_size, shuffle = False, pin_memory = True)

params = AdaroundParameters(data_loader=val_dataloader, num_batches=4, default_num_iterations=32,
                            default_reg_param=0.01, default_beta_range=(20, 2))

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape).cuda()


# Returns model with adarounded weights and their corresponding encodings
adarounded_model = Adaround.apply_adaround(prepared_model, dummy_input, params, path='./',
                                            filename_prefix='resnet18', default_param_bw=8,
                                            default_quant_scheme=QuantScheme.post_training_tf,
                                            default_config_file="/home/anaconda3/envs/aimet_py1.20/lib/python3.6/site-packages/aimet_common/quantsim_config/htp_quantsim_config.json")

sim = QuantizationSimModel(adarounded_model, quant_scheme=QuantScheme.post_training_tf_enhanced, default_param_bw=8,
                    default_output_bw=8, dummy_input=dummy_input)

# Set and freeze encodings to use same quantization grid and then invoke compute encodings
sim.set_and_freeze_param_encodings(encoding_path='./resnet18.encodings')

sim.compute_encodings(model_eval("/home/rayari/5000/", image_size = (1,3,224,224), batch_size=64, num_workers=16, model_name='PTQ'),(1,True))