# 运行 ONNX 模型，保存输入输出
polygraphy run \
onnx_zoo/hxmask.onnx \
--onnxrt \
--input-shapes imgs:[1,3,256,256] \
--val-range [0,1] \
--save-inputs onnx_inputs.json \
--save-outputs onnx_outputs.json
# 运行 TRT 模型，载入 ONNX 输入输出，对比输出的相对误差与绝对误差
polygraphy run \
hxmask.plan \
--model-type engine \
--trt \
--load-inputs onnx_inputs.json \
--load-outputs onnx_outputs.json \
--rtol 1e-3 \
--atol 1e-3


