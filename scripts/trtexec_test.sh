trtexec \
    --onnx=onnx_zoo/hxmask.onnx\
    --workspace=20480 \
    --saveEngine=model-FP32.plan \
    --shapes=imgs:1x3x256x256  \
    --useSpinWait \
    --verbose \
    > hx.txt