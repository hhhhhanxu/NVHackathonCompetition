import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from collections import OrderedDict
import sys
sys.path.append('/root/hx/NVHackathonCompetition/')
from utils.print_color_txt import colorstr


def surgeon(onnx_path):
    # 读取 .onnx 并进行调整
    graph = gs.import_onnx(onnx.load(onnx_path))
    print(colorstr('加载模型:')+onnx_path,"节点数量:",colorstr('red',str(len(graph.nodes))))
    # 从slice45开始往下找

    FirstLayerNormNode = None
    ConvNode = None

    nSTReshape = 0
    nSTReshapeRoll = 0

    # ------------------------------------------------------end
    for node_id, node in enumerate(graph.nodes):
        if node.name == 'LayerNorm_1':
            FirstLayerNormNode = node
        if node.name == 'Conv_2054':
            ConvNode = node
    for node_id, node in enumerate(graph.nodes):
        # without shift
        if node.op == "LayerNorm" and node.outputs[0].name != "outputs" and \
                node.o().op == "Reshape" and node.o().o().op != "Slice" and node.o().o(4).op == "Reshape":
            reshapeN = node.o()
            LastN = node.o().o(4)
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":0, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        # without shift 可不用
        if node.op == "Reshape" and len(node.outputs) > 0  and node.outputs[0].name != "outputs" and node.o().op == "Reshape" and \
                node.o().o().op == "Add" and node.o().o().o(1).op == "LayerNorm":
            reshapeN = node
            LastN = node.o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":1, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        # shift
        if node.op == "LayerNorm" and node.outputs[0].name != "outputs" and node.o().op == "Reshape"  and \
                len(node.o().outputs) > 0 and node.o().o().op == "Slice" and \
                node.o().o().o().op == "Concat" and node.o().o().o().o().op == "Slice" and \
                node.o().o().o().o().o().op == "Concat" and node.o().o().o().o().o().o(4).op == "Reshape":
            reshapeN = node.o()
            LastN = node.o().o().o().o().o().o(4)
            STReshapeRollN = gs.Node("STReshapeRoll", "STReshapeRoll-" + str(nSTReshapeRoll), 
                                    inputs=[reshapeN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":0, "window_size":8, "shift": -4})
            graph.nodes.append(STReshapeRollN)
            nSTReshapeRoll += 1
            LastN.outputs = []

        # shift
        if node.op == "Reshape" and len(node.outputs) > 0 and node.outputs[0].name != "outputs" and node.o().op == "Slice" and \
                node.o().o().op == "Concat" and node.o().o().o().op == "Slice" and \
                node.o().o().o().o().op == "Concat" and node.o().o().o().o().o().op == "Reshape" and \
                node.o().o().o().o().o().o().op == "Add" and node.o().o().o().o().o().o().o(1).op == "LayerNorm":
            reshapeN = node
            LastN = node.o().o().o().o().o()
            STReshapeRollN = gs.Node("STReshapeRoll", "STReshapeRoll-" + str(nSTReshapeRoll), 
                                    inputs=[reshapeN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":1, "window_size":8, "shift": 4})
            graph.nodes.append(STReshapeRollN)
            nSTReshapeRoll += 1
            LastN.outputs = []

    for node_id, node in enumerate(graph.nodes):
        # 可不用
        if node.op == "Transpose" and node.o().op == "Reshape" and \
            len(node.o().outputs) > 0 and node.o().outputs[0].name != "outputs" and \
            node.o().o().op == "Reshape":
            FirstN = node.o()
            LastN = node.o().o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[FirstN.inputs[0], FirstLayerNormNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":2, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

        if node.op == "Reshape" and len(node.outputs) > 0 and node.outputs[0].name != "outputs" and \
                node.o().op == "Reshape" and len(node.o().outputs) > 0 and node.o().o().op == "Transpose":
            FirstN = node
            LastN = node.o()
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[FirstN.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[LastN.outputs[0]],
                                    attrs={"type":3, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            LastN.outputs = []

    graph.cleanup().toposort()

    for node_id, node in enumerate(graph.nodes):
        if node.op == "Reshape" and node.o().op == "Conv":
            STReshapeN = gs.Node("STReshape", "STReshape-" + str(nSTReshape), 
                                    inputs=[node.inputs[0], ConvNode.outputs[0]], 
                                    outputs=[node.outputs[0]],
                                    attrs={"type":4, "window_size":8})
            graph.nodes.append(STReshapeN)
            nSTReshape += 1
            node.outputs = []

    print('完成'+colorstr('red',str(nSTReshape))+'个Reshape节点的替换')
    print('完成'+colorstr('red',str(nSTReshapeRoll))+'个Roll节点的替换')

    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_reshape_roll.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(colorstr("新模型节点数:") ,colorstr('red',str(len(graph.nodes))) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/SwinIR_LN_mask_Gelu.onnx",
                        help="onnx file path.")
    args = parser.parse_args()
    surgeon(args.onnxFile)
