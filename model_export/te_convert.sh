model_transform.py  \
        --model_name sdv15_te  \
        --input_shape [[1,77]] \
        --model_def path_to_text_encoder_onnx \
        --mlir sd15_te.mlir

model_deploy.py\
        --mlir sd15_te.mlir \
        --quantize F32 \
        --chip bm1684x \
        --model path_to_text_encoder.bmodel
