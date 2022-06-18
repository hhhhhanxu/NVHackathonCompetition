trtexec \
    --onnx=./onnx_zoo/calculate_mask_head_surgeon.onnx \
    --minShapes=imgs:1x3x256x256 \
    --optShapes=imgs:4x3x256x256 \
    --maxShapes=imgs:16x3x256x256 \
    --workspace=10240 \
    --saveEngine=model-FP32.plan \
    --shapes=imgs:4x3x256x256 \
    --plugins=./plugin/WindowsMaskPlugin.so
