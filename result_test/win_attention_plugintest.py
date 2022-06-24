import sys
sys.path.append('/root/hx/NVHackathonCompetition/')

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple

from network_swinir_simplify import WindowAttention

class winatten(nn.Module):
    def __init__(self,dim,window_size,num_heads,qkv_bias,qk_scale,attn_drop,drop) -> None:
        super().__init__()
        self.window_size= window_size
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self,x,mask=None):
        x = self.attn(x, mask=mask)


        return x

model = winatten(dim=60,window_size=8,num_heads=6,qkv_bias=True,qk_scale=None,attn_drop=0.,drop=0.)
model.cuda()
onnx_file_name = 'win_attn.onnx'

x = torch.randn((1089,64,60), requires_grad=False).cuda()

torch.onnx.export(model = model,args = (x),f=onnx_file_name,
                do_constant_folding=True,
                opset_version=13, # 13版本
                verbose=False,
                input_names=['inputs'],
                output_names=['outputs'],
                dynamic_axes={
                    'inputs':{0: 'batch'},
                    'outputs':{0: 'batch'}
                })

