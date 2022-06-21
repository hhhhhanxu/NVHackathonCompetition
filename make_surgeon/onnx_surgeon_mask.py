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
    ConstantOfShapeNode = None
    ShapeNode = None
    ScatterNDNode = None

    nWindowsMask = 0
    # ------------------------------------------------------添加shift_window 的plugin
    for node_id, node in enumerate(graph.nodes):

        if node.name == "ConstantOfShape_1026": # ConstantOfShape_973
            ConstantOfShapeNode = node
        if node.name == "Shape_1072": # Shape_84 Shape_139
            ShapeNode = node
        if node.name == "ScatterND_1989": # ScatterND_1025 ScatterND_1080
            ScatterNDNode = node

    if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
        img_mask = ConstantOfShapeNode.outputs[0]
        img_mask_shape = ShapeNode.outputs[0]
        WindowsMaskN = gs.Node("WindowsMask", "WindowsMask_" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])

        WindowsMaskN.attrs = OrderedDict(
            window_size = np.array([8],dtype=np.int32),
            shift_size = np.array([4],dtype=np.int32),
            plugin_version = "1",
            plugin_namespace = ""
        )

        graph.nodes.append(WindowsMaskN)
        nWindowsMask += 1
        ScatterNDNode.outputs = []
    # ------------------------------------------------------添加不带shift 的plugin
    ConstantOfShapeNode = None
    ShapeNode = None
    ScatterNDNode = None

    for node in graph.nodes:
        if node.name == "ConstantOfShape_57": 
            ConstantOfShapeNode = node
        if node.name == "Shape_103":
            ShapeNode = node
        if node.name == "ScatterND_960": # ScatterND_1025 ScatterND_1080
            ScatterNDNode = node

    if ConstantOfShapeNode is not None and ShapeNode is not None and ScatterNDNode is not None:
        img_mask = ConstantOfShapeNode.outputs[0]
        img_mask_shape = ShapeNode.outputs[0]

        WindowsMaskN = gs.Node("WindowsMask", "WindowsMask_" + str(nWindowsMask), inputs=[img_mask, img_mask_shape], outputs=[ScatterNDNode.outputs[0]])
        WindowsMaskN.attrs = OrderedDict(
            window_size = np.array([8],dtype=np.int32),
            shift_size = np.array([0],dtype=np.int32),
            plugin_version = "1",
            plugin_namespace = ""
        )
        graph.nodes.append(WindowsMaskN)
        nWindowsMask += 1
        ScatterNDNode.outputs.clear()
    # ------------------------------------------------------end
    print(f"nWindowsMask: {nWindowsMask}")


    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_surgeon.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(colorstr("surgeon model nodes:") ,len(graph.nodes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/calculate_mask.onnx",
                        help="onnx file path.")
    args = parser.parse_args()
    surgeon(args.onnxFile)
