import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from collections import OrderedDict

from torch import scatter_reduce

def surgeon(onnx_path):
    # 读取 .onnx 并进行调整
    graph = gs.import_onnx(onnx.load(onnx_path))

    ConstantOfShapeNode = None
    ShapeNode = None
    ScatterNDNode = None


    nWindowsMask = 0

    for node_id, node in enumerate(graph.nodes):

        if node.name == "ConstantOfShape_973": # ConstantOfShape_62 ConstantOfShape_117
            ConstantOfShapeNode = node
        if node.name == "Shape_1019": # Shape_84 Shape_139
            ShapeNode = node
        if node.name == "ScatterND_1936": # ScatterND_1025 ScatterND_1080
            ScatterNDNode = node
        # if node.name == "Conv_50":
        #     ConvNode = node
        # if node.name == "Reshape_99": # Reshape_2521 Reshape_99
        #     ReshapeNode = node


    if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
        img_mask = ConstantOfShapeNode.outputs[0]
        img_mask_shape = ShapeNode.outputs[0]
        WindowsMaskN = gs.Node("WindowsMask", "WindowsMask_" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])
        WindowsMaskN.attrs = OrderedDict([
            ["window_size",8],
            ["shift_size",4],
        ])

        graph.nodes.append(WindowsMaskN)
        nWindowsMask += 1
        ScatterNDNode.outputs = []
    print(ConstantOfShapeNode,ShapeNode,ScatterNDNode)
    print(f"nWindowsMask: {nWindowsMask}")


    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_surgeon.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(f"surgeon model nodes: {len(graph.nodes)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/calculate_mask_first.onnx",
                        help="onnx file path.")
    args = parser.parse_args()
    surgeon(args.onnxFile)
