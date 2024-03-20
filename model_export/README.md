# 将任何StableDiffusion1.5的模型转化为一秒出图的bmodel

## 1. 模型trace

### 环境配置
```sh
python      3.10 
torch       1.12.0
torchvision 0.13.0 
diffusers   0.24.0
```
### 命令
- trace UNet / text encoder
    ```sh
    python export_lcm.py \
    --safetensors_path path_to_your_safetensors \
    --lcm_lora_path path_to_lcm_lora_model_dir \
    --unet_pt_path output_Unet_pt_path  \
    --text_encoder_onnx_path output_text_encoder_onnx_path
    ```

- trace VAE encoder / decoder
    ```sh
    python export_vae.py \
    -s path_to_your_safetensors \
    -p output_Unet_pt_path
    ```

## 2. 转换bmodel

### 环境配置
```sh
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest`
```
详见[tpu-mlir文档中的环境配置章节](https://tpumlir.org/docs/quick_start/02_env.html)。

### 在docker内执行命令
注意：在sh脚本中修改作为输入的pt/onnx文件路径和输出文件的路径。
- 转换UNet pt为bmodel：`bash unet_convert.sh`
- 转换Text encoder ONNX为bmodel：`bash te_convert.sh`
- 转换VAE pt为bmodel：`bash vae_convert.sh`, 按照需要修改`encoder_shapes`和`decoder_shapes`。
    
    `image size`必须是8的整数倍，相应的`latent shape`是`[1,4,size[0]//8,size[1]//8]`。
## 3. 使用bmodel
在`model_path.py`中模型列表的最前面填上新模型的路径信息（应用默认加载该列表中的第一个模型）。

## 常见问题
1. 转换mlir时，报错scale_dot_product_attenton算子不支持。
    
    Solution：trace模型时使用的torch版本较高，trace出的pt模型中包含了该算子；使用较低版本的torch（如1.12.0）可以规避这个问题。
2. 转换bmodel后，出图明显异常。

    Solution：使用较低版本的torch（如1.12.0）重新跑`model_export`中的trace脚本，再重新转bmodel。
3. bmodel出图结果make sense，但与纯CPU上跑出的结果不完全一致。
    
    Solution：检查text encoder输出差异是否过大。C站上SD1.5的不同checkpoint，Unet和text encoder都不同，都需要重新trace和转换bmodel！另外，可比对F32、F16、BF16的输出差异。
