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
    ConstantOfShapeNode = None
    ShapeNode = None
    ScatterNDNode = None

    nWindowsMask = 0
    # ------------------------------------------------------添加shift_window 的plugin
    for node_id, node in enumerate(graph.nodes):

        if node.name == "ConstantOfShape_1032": 
            ConstantOfShapeNode = node
        if node.name == "Shape_1078": 
            ShapeNode = node
        if node.name == "ScatterND_1995": 
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
        if node.name == "ConstantOfShape_63": 
            ConstantOfShapeNode = node
        if node.name == "Shape_109":
            ShapeNode = node
        if node.name == "ScatterND_966": 
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
    print('完成'+colorstr('red',str(nWindowsMask))+'个WindowsMask节点的替换')
    # print(f"nWindowsMask: {nWindowsMask}")

    graph.cleanup().toposort()
    surgeon_onnx_path = onnx_path.replace(".onnx", "_mask.onnx")
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print(colorstr("新模型节点数:") ,colorstr('red',str(len(graph.nodes))) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/SwinIR_LN.onnx",
                        help="onnx file path.")
    args = parser.parse_args()
    surgeon(args.onnxFile)
