import torch 
import sys
import time
sys.path.append('./SwinIR')

import numpy as np
import cv2 
import onnx
import onnx_graphsurgeon as gs

# from models.network_swinir import SwinIR as net    # 这个是原始模型
from network_swinir_simplify import SwinIR as net

checkpoint = torch.load("./SwinIR/model_zoo/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth")

model = net(upscale=2, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
param_key_g = 'params'

model.load_state_dict(checkpoint[param_key_g] if param_key_g in checkpoint.keys() else checkpoint)
model.eval() #重要！
model.cuda()

window_size = 8
N = 1
C = 3
W = 32 * window_size 
H = 32 * window_size 

onnx_version = "hx_change_mask"
onnx_file_name = './onnx_zoo/{}.onnx'.format(onnx_version)

x = torch.randn((N,C,W,H), requires_grad=False).cuda()
torch.onnx.export(model = model,args = (x),f=onnx_file_name,
                do_constant_folding=True,
                opset_version=13, # 13版本
                verbose=False,
                input_names=['imgs'],
                output_names=['outputs'],
                dynamic_axes={
                    'imgs':{0: 'batch', 2: 'height', 3: 'width'},
                    'outputs':{0: 'batch', 2: 'height', 3: 'width'}
                })
# 大概吃9个G
model = onnx.load(onnx_file_name)
graph = gs.import_onnx(model)

print("导出{}onnx模型节点数:{}".format(onnx_file_name,len(graph.nodes)))
