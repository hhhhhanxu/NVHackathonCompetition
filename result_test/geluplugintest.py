import torch
import torch.nn as nn

import onnx
import onnx_graphsurgeon as gs

class test_gelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act_layer = nn.GELU()

    def forward(self,x):
        x = self.act_layer(x)
        x = self.act_layer(x)
        x = self.act_layer(x)
        return x
    

model = test_gelu().cuda()
x = torch.randn((1,3,256,256)).cuda()
y = model(x)
# 导出模型
window_size = 8
N = 1
C = 3
W = 32 * window_size 
H = 32 * window_size 

onnx_file_name = 'gelu.onnx'

x = torch.randn((N,C,W,H), requires_grad=False).cuda()
torch.onnx.export(model = model,args = (x),f=onnx_file_name,
                do_constant_folding=True,
                opset_version=13, # 13版本
                verbose=False,
                input_names=['imgs'],
                output_names=['outputs'],
                dynamic_axes={
                    'imgs':{0: 'batch',2:'w',3:'h'},
                    'outputs':{0: 'batch',2:'w',3:'h'}
                })
# 修改onnx模型
# onnx_model = onnx.load(onnx_file_name)
# graph = gs.import_onnx(onnx_model)
# print('原始节点数:',len(graph.nodes))
# new_node = gs.Node(op='Gelu',name='Gelu_1',inputs=[graph.inputs[0]],outputs=[graph.outputs[0]])
# graph.nodes.append(new_node)
# for node in graph.nodes:
#     if node.name == 'Mul_7':
#         node.outputs.clear()
# graph.cleanup()
# onnx.save(gs.export_onnx(graph),'gelu_surgeon.onnx')