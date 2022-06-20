import argparse
from glob import glob
from pickle import NONE
import tensorrt as trt
import ctypes

def onnx2trt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="./onnx_zoo/calculate_mask_head_surgeon.onnx",
                        help="onnx file path.")
    parser.add_argument("--trtFile", type=str, default=None,
                        help="onnx file path.")
    args = parser.parse_args()

    onnxFile = args.onnxFile
    trtFile = args.trtFile
    if trtFile is None:
        trtFile = onnxFile.replace(".onnx", ".plan")

    print(f"onnxFile: {onnxFile}")
    print(f"trtFile: {trtFile}")
    
    # 获取plugin列表
    PluginPath   = "/root/hx/NVHackathonCompetition/plugin/"
    soFileList = glob(PluginPath + "*/*.so")
    print("加载plugin:",soFileList)
    # 加载plugin
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)  # 基于ctypes库

    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnxFile)  #
    config = builder.create_builder_config()
    config.max_workspace_size = 12 << 30  # 12G
    
    config.flags = 1 << int(trt.BuilderFlag.FP16) # 开启FP16？

    profile = builder.create_optimization_profile()
    print("==== inputs name:")
    for i in range(1):
        print(f"Input{i} name: ", network.get_input(i).name)
    inputTensor1 = network.get_input(0)

    profile.set_shape(inputTensor1.name, [1, 3, 256, 256], [4, 3, 256, 256], [16, 3, 256, 256]) # 动态维度
    config.add_optimization_profile(profile)

    config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
    config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=False)
    # config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # config.clear_flag(trt.BuilderFlag.TF32)

    engineString = builder.build_serialized_network(network, config)

    try:
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        print("export .plan successful")
    except:
        print("export .plan fail")
    # 将没法转换的子图单独保存 

if __name__ == '__main__':
    onnx2trt()