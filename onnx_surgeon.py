import onnx
import onnx_graphsurgeon as gs

onnx_file_name = 'onnx_zoo/LayerNorm.onnx'
onnx_model = onnx.load(onnx_file_name)
graph = gs.import_onnx(onnx_model)

print('load model {}, the nodes number is {}'.format(onnx_file_name,len(graph.nodes)))

# ------------------------------------------------------------
# 替换掉LN节点
LayerNorm_N = 0
for node in graph.nodes:
    # if node.op == 'ReduceMean':
    #     print(node.o().o(0))  # 0是Pow 1是Div
    #     print(node.o().o(1))  # 0是Pow 1是Div
    #     assert 0
    if node.op == 'ReduceMean' and node.o().op=='Sub' \
        and node.o().o(0).op == 'Pow' and node.o().o(0).o().op == 'ReduceMean' \
            and node.o().o(0).o().o().op == 'Add' and node.o().o(0).o().o().o().op == 'Sqrt' \
                and node.o().o(0).o().o().o().o().op == 'Div' \
        and node.o().o(1).op == 'Div' and node.o().o(1) == node.o().o(0).o().o().o().o() \
            and node.o().o(1).o().op == 'Mul' and node.o().o(1).o().o().op == 'Add':
            # 找到LN节点
            LayerNorm_N+=1
            end_node = node.o().o(1).o().o()
            
            new_node = gs.Node(op='LayerNorm',name='LayerNorm_{}'.format(LayerNorm_N))
            new_node.inputs = [node.inputs[0]]
            new_node.outputs = [end_node.outputs[0]]
            print(LayerNorm_N)
            graph.nodes.append(new_node)
            end_node.outputs.clear()

graph.cleanup()
onnx.save(gs.export_onnx(graph),'test.onnx')

            
