model_transform.py  \
	--model_name unet_no_cfg  \
	--input_shape [[1,4,64,64],[1],[1,77,768],[1,1280,8,8],[1,320,64,64],[1,320,64,64],[1,320,64,64],[1,320,32,32],[1,640,32,32],[1,640,32,32],[1,640,16,16],[1,1280,16,16],[1,1280,16,16],[1,1280,8,8],[1,1280,8,8],[1,1280,8,8]] \
	--model_def awportrait_v13_unet_lcm_patch1.pt \
    --mlir unet_2_512.mlir

model_deploy.py\
    --mlir unet_2_512.mlir \
	--quantize BF16 \
    --chip bm1684x \
    --model unet_2_1684x_BF16.bmodel