# 512x512
model_transform.py  \
	--model_name unet_no_cfg  \
	--input_shape [[1,4,64,64],[1],[1,77,768],[1,1280,8,8],[1,320,64,64],[1,320,64,64],[1,320,64,64],[1,320,32,32],[1,640,32,32],[1,640,32,32],[1,640,16,16],[1,1280,16,16],[1,1280,16,16],[1,1280,8,8],[1,1280,8,8],[1,1280,8,8]] \
	--model_def path_to_unet_pt \
    --mlir unet_no_cfg.mlir

# 768x768
#model_transform.py  \
#	--model_name unet_no_cfg  \
#	--input_shape [[1,4,96,96],[1],[1,77,768],[1,1280,12,12],[1,320,96,96],[1,320,96,96],[1,320,96,96],[1,320,48,48],[1,640,48,48],[1,640,48,48],[1,640,24,24],[1,1280,24,24],[1,1280,24,24],[1,1280,12,12],[1,1280,12,12],[1,1280,12,12]] \
#	--model_def path_to_unet_pt \
#    --mlir unet_no_cfg.mlir

# quantize 可选BF16、F16和F32
model_deploy.py\
    --mlir unet_no_cfg.mlir \
	--quantize BF16 \
    --chip bm1684x \
    --model path_to_unet_no_cfg.bmodel
