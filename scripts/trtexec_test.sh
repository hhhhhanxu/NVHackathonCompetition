trtexec \
    --loadEngine=./LayerNorm_surgeonv1.plan \
    --plugins=./plugin/LayerNorm/LayerNormPlugin.so \
    --shapes=imgs:1x3x256x256 \
