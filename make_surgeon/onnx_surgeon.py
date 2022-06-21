import onnx
import onnx_graphsurgeon as gs
import numpy as np

from collections import OrderedDict

import sys
sys.path.append('/root/hx/NVHackathonCompetition/')
from utils.print_color_txt import colorstr

onnx_file_name = 'onnx_zoo/SwinIR.onnx'
onnx_model = onnx.load(onnx_file_name)
graph = gs.import_onnx(onnx_model)

print(colorstr('加载模型:')+onnx_file_name,"节点数量:",colorstr('red',str(len(graph.nodes))))

# ------------------------------------------------------------
# 替换掉LN节点
LayerNorm_N = 0
for node in graph.nodes:
    # if node.op == 'ReduceMean':
    #     print(node.o().o(0))  # 0是Pow 1是Div
    #     print(node.o().o(1))  # 0是Pow 1是Div
    #     assert 0
    if node.op == 'ReduceMean' and node.o().op=='Sub' \
        and node.o().o(0).op == 'Pow' \
        and node.o().o(0).o().op == 'ReduceMean' \
        and node.o().o(0).o().o().op == 'Add' \
        and node.o().o(0).o().o().o().op == 'Sqrt' \
        and node.o().o(0).o().o().o().o().op == 'Div' \
        and node.o().o(1).op == 'Div' \
        and node.o().o(1) == node.o().o(0).o().o().o().o() \
        and node.o().o(1).o().op == 'Mul' and node.o().o(1).o().o().op == 'Add':
            # 找到LN节点
            LayerNorm_N+=1
            mul_node = node.o().o(1).o()
            add_node = node.o().o(1).o().o()
            # print(mul_node.inputs)  # 对于这种乘法和加法节点，其中是带有常量参数的，作为输入的其中一个
            # print(add_node.inputs)

            end_node = node.o().o(1).o().o()
            
            new_node = gs.Node(op='LayerNorm',name='LayerNorm_{}'.format(LayerNorm_N))
            new_node.inputs = [node.inputs[0],mul_node.inputs[1],add_node.inputs[1]]  # 分别对应plugin里的gamma和beta
            new_node.outputs = [end_node.outputs[0]]
            # new_node.attrs = OrderedDict([['nHiddenDimension',96]])  # 这个写法貌似有点问题
            new_node.attrs = OrderedDict(
                                nHiddenDimension = np.array([60],dtype=np.int32), # plugin的初始化参数，是对应模型中的embed_dim=60
                                plugin_version = "1",
                                plugin_namespace = ""
            )

            # print(LayerNorm_N)
            graph.nodes.append(new_node)
            end_node.outputs.clear()

print('完成'+colorstr('red',str(LayerNorm_N))+'个LayerNorm节点的转换')
graph.cleanup()
print(colorstr('新模型节点数:'),colorstr('red',str(len(graph.nodes))) )
output_name =  onnx_file_name.split('.')[0]+'_LN.onnx'
onnx.save(gs.export_onnx(graph),output_name)
