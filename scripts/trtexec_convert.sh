trtexec \
    --onnx=./onnx_zoo/SwinIR_LN_mask_Gelu.onnx\
    --workspace=20480 \
    --saveEngine=SwinIR.plan \
    --shapes=imgs:1x3x256x256  \
    --useSpinWait \
    --plugins=./plugin/LayerNormPlugin-V2.2-CUB-TRT8/LayerNormPlugin.so \
    --plugins=./plugin/WindowsMaskPlugin-V2-HX/WindowsMaskPlugin.so \
    --plugins=./plugin/GeluPlugin-V1-HX/GeluPlugin.so\

