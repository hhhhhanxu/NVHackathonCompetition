import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from collections import OrderedDict

from utils.print_color_txt import colorstr

def surgeon(onnx_path):
    # 读取 .onnx 并进行调整
    graph = gs.import_onnx(onnx.load(onnx_path))
    print(colorstr("original model nodes:"),len(graph.nodes))

    nWindowsFill = 0

    for node in graph.nodes:
        if node.op == 'Sub' and node.i().op == 'Unsqueeze' :
            if node.o(0).op == 'Equal' and node.o(1).op == 'Equal' and node.o(2).op == 'Where'\
                and node.o(0).o().op=='Not' and node.o(0).o().o().op=='Cast' \
                and node.o(0).o().o().o() == node.o(2) \
                and node.o(2).o().op == 'Where' and node.o(1).o().op=='Cast' \
                and node.o(2).o() == node.o(1).o().o() \
                and node.o(2).o().o().op == 'Cast':

                inputs = node.outputs[0]
                outputs = node.o(2).o().o().outputs[0]

                nWindowsFill +=1
                
                newFillNode = gs.Node(op='WindowsFill',name=f"WindowsFill_{nWindowsFill}",inputs=[inputs],outputs=[outputs])
                node.o(2).o().o().outputs.clear()
                graph.nodes.append(newFillNode)
    

                
    print(f"nWindowsFill: {nWindowsFill}")


    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_surgeon.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(f"surgeon model nodes: {len(graph.nodes)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/calculate_mask_surgeon_1.onnx",
                        help="onnx file path.")
    args = parser.parse_args()
    surgeon(args.onnxFile)
