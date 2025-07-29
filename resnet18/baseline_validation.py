import random
import numpy as np
import torch
import time
import torch.nn as nn
import copy
import torchvision as tv
import torchvision.models as models

import os 
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from utils import AverageMeter, accuracy
import argparse
import onnxruntime as onnxrt
import PIL

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


def arguments():
	parser = argparse.ArgumentParser(description='Evaluation script for PyTorch ImageNet networks.')

	parser.add_argument('--model-path',             	help='Path to checkpoint directory to load from.', default = "./mode/mv2qat_modeldef.pth", type=str)
	parser.add_argument('--images-dir',         		help='Imagenet eval image', default='./ILSVRC2012/', type=str)
	parser.add_argument('--seed',						help='Seed number for reproducibility', type = int, default=0)
	
	parser.add_argument('--quant-tricks', 				help='Preprocessing prior to Quantization', choices=['BNfold', 'CLS', 'HBF', 'CLE', 'BC', 'adaround'], nargs = "+")
	parser.add_argument('--quant-scheme',               help='Quant scheme to use for quantization (tf, tf_enhanced, range_learning_tf, range_learning_tf_enhanced).', default='tf', choices = ['tf', 'tf_enhanced', 'range_learning_tf', 'range_learning_tf_enhanced'])
	parser.add_argument('--round-mode',                 help='Round mode for quantization.', default='nearest')
	parser.add_argument('--default-output-bw',          help='Default output bitwidth for quantization.', type = int, default=8)
	parser.add_argument('--default-param-bw',           help='Default parameter bitwidth for quantization.', type = int, default=8)
	parser.add_argument('--config-file',       			help='Quantsim configuration file.', default=None, type=str)
	parser.add_argument('--cuda',						help='Enable cuda for a model', default=True)
	
	parser.add_argument('--batch-size',					help='Data batch size for a model', type = int, default=4)


	args = parser.parse_args()
	return args

def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main():
    args = arguments()
    seed(args)
    print("Pytorch model list:\n ", dir(tv.models))
    print("\nLoad pretrained model -> resnet18")
    model = tv.models.resnet18(pretrained=True)
    model.eval()
    input_shape = (1,3,224,224)
    image_size = input_shape[-1]

    eval_func = model_eval("/home/rayari/5000/", image_size, batch_size=args.batch_size, num_workers=16, model_name='Baseline model')
    print(eval_func(model.cuda(), (1, True)))
    
    # # Execute JIT trace.
    # example = torch.randn(1, 3, 224, 224).cuda()
    # traced_script_module = torch.jit.trace(model, example)
    # # Save the TorchScript model
    # traced_script_module.save("/media/DATA2/rayari/5000//resnet18/traced_resnet18_model.pt")
    # #traced_model = torch.jit.load("/media/DATA2/rayari/resnet18/traced_resnet18_model.pt")
    # print("\nJIT trace is executed successfully.")
    
    # # ONNX process.
    # # Export to ONNX model.
    # dummy_input = torch.randn(1, 3, 224, 224).cuda()
    # input_names = [ "actual_input" ]
    # output_names = [ "output" ]
    # torch.onnx.export(model, 
    #                   dummy_input,
    #                   "/media/DATA2/rayari/resnet18/resnet18.onnx",
    #                   verbose=False,
    #                   input_names=input_names,
    #                   output_names=output_names,
    #                   export_params=True,
    #                   example_outputs=torch.rand((1, 1000))
    #                   )

    # print("\nExported ONNX model successfully.\n")
    

    # import aimet_torch
    # from aimet_common.defs import QuantScheme
    # # from aimet_torch import cross_layer_equalization
    # from aimet_torch.quantsim import QuantizationSimModel
    # # import aimet_torch
    # from aimet_torch.cross_layer_equalization import equalize_model
    # # from aimet_torch import batch_norm_fold
    
    # print("Enter the model into the PTQ techniques.\n")
    # # batch_norm_fold.fold_all_batch_norms(model, input_shape)
    # equalize_model(model, input_shape)
    # print("Equalize the model.\n")
    # #cross_layer_equalization.CrossLayerScaling.scale_model(model, input_shape)

    # sim = QuantizationSimModel(model=model.cuda(),default_output_bw=8,
    #                             default_param_bw=8, dummy_input=torch.rand(1,3,224,224).cuda(),
    #                            quant_scheme = QuantScheme.post_training_tf_enhanced,config_file="/home/anaconda3/envs/aimet_py1.20/lib/python3.6/site-packages/aimet_common/quantsim_config/htp_quantsim_config.json") 
    # print("Quantsim the model.\n")
    # # #post_quant_top1 = _50k_eval_func(sim.model.cuda(), (1, True))
    # # sim.compute_encodings(_5000_eval_func, (1, True))
    # print("Compute encoding.\n")
    # opt_eval_func = model_eval("/home/rayari/5000/", image_size, batch_size=args.batch_size, num_workers=16, model_name='PTQ model')
    # sim.compute_encodings(opt_eval_func, (1, True))
        
    # # post_quant_top1 = _2000_eval_func(sim.model.cuda(), (1, True))
    # # # print("Post Quant Top1 :", post_quant_top1)
  
    # sim.export(path="/media/DATA2/rayari/resnet18/ptq_models/", filename_prefix='resnet18_quantization', dummy_input=torch.rand(1,3,224,224))
    # print("Quantized Model exported successfully.\n")
    
if __name__ == '__main__':
    main()