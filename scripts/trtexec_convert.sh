trtexec \
    --onnx=./onnx_zoo/calculate_mask_first_surgeon.onnx\
    --workspace=20480 \
    --saveEngine=model-FP32.plan \
    --shapes=imgs:1x3x256x256  \
    --useSpinWait \
    --plugins=./plugin/WindowsMaskPlugin.so \
