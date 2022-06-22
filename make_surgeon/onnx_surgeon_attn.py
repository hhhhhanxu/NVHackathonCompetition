import onnx 
import onnx_graphsurgeon as gs

import sys
sys.path.append('/root/hx/NVHackathonCompetition/')

from utils.print_color_txt import colorstr

onnx_file_name = "onnx_zoo/SwinIR_LN_mask_Gelu.onnx"
onnx_model = onnx.load(onnx_file_name)
graph = gs.import_onnx(onnx_model)

print(colorstr('加载模型:')+onnx_file_name,"节点数量:",colorstr('red',str(len(graph.nodes))))


Attn_N = 0
for node in graph.nodes:
    if node.op=='Softmax' and node.o().op == 'MatMul' and node.o().o().op== 'Transpose' and node.o().o().o().op=='Reshape' and node.o().o().o().o().op=='MatMul' and node.o().o().o().o().o().op=='Add':
        if node.i().op == 'Reshape' and node.i().i(0).op=='Add' and node.i().i(0).i(0).op=='Reshape' and node.i().i(0).i(0).i(0).op=='Add' and node.i().i(0).i(0).i(0).i().op=='MatMul': 
            if node.i().i(0).i(0).i(0).i().i().i().i().op == 'Transpose' and node.i().i(0).i(0).i(0).i().i().i().i().i().op == 'Reshape' and \
            node.i().i(0).i(0).i(0).i().i().i().i().i().i(1).op=='Concat' and node.i().i(0).i(0).i(0).i().i().i().i().i().i(1).i().op=='Unsqueeze' and \
            node.i().i(0).i(0).i(0).i().i().i().i().i().i(1).i().i().op == 'Gather' and node.i().i(0).i(0).i(0).i().i().i().i().i().i(1).i().i().i().op=='Shape' and \
            node.i().i(0).i(0).i(0).i().i().i().i().i().i(1).i().i().i().i().op == 'Reshape':

                Attn_N += 1

                start_node = node.i().i(0).i(0).i(0).i().i().i().i().i().i(1).i().i().i().i()
                end_node = node.o().o().o().o().o()
                new_node = gs.Node(op='Attn',name='Attn_{}'.format(Attn_N),inputs=[start_node.outputs[0]],outputs=[end_node.outputs[0]])

                end_node.outputs.clear()
                graph.nodes.append(new_node)

    # if node.name == 'MatMul_2156':
    #     print(node.i())
    #     print(node.o())


print('完成'+colorstr('red',str(Attn_N))+'个Attention节点的转换')
graph.cleanup()
print(colorstr('新模型节点数:'),colorstr('red',str(len(graph.nodes))) )

output_name =  onnx_file_name.split('.')[0]+'_Attn.onnx'
onnx.save(gs.export_onnx(graph),output_name)
