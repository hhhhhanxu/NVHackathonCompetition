import onnx
import onnx_graphsurgeon as gs
import numpy as np

from collections import OrderedDict

import sys
sys.path.append('/root/hx/NVHackathonCompetition/')
from utils.print_color_txt import colorstr

onnx_file_name = 'onnx_zoo/SwinIR_LN_mask.onnx'

onnx_model = onnx.load(onnx_file_name)
graph = gs.import_onnx(onnx_model)

print(colorstr('加载模型:')+onnx_file_name,"节点数量:",colorstr('red',str(len(graph.nodes))))

# ------------------------------------------------------------
# 替换掉LN节点
Gelu_N = 0
for node in graph.nodes:
    if node.op == 'Div' and node.o().op=='Erf' and node.o().o().op=='Add' and node.o().o().o().op=='Mul' \
        and node.o().o().o().o().op=='Mul':
            # 找到LN节点
            Gelu_N+=1
            end_node = node.o().o().o().o()

            new_node = gs.Node(op='Gelu',name='Gelu_{}'.format(Gelu_N),inputs=[node.inputs[0]],outputs=[end_node.outputs[0]])
            # print(LayerNorm_N)
            graph.nodes.append(new_node)
            end_node.outputs.clear()


print('完成'+colorstr('red',str(Gelu_N))+'个Gelu节点的转换')
graph.cleanup()
print(colorstr('新模型节点数:'),colorstr('red',str(len(graph.nodes))) )
output_name =  onnx_file_name.split('.')[0]+'_Gelu.onnx'
onnx.save(gs.export_onnx(graph),output_name)
