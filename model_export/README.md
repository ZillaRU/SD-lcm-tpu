# 将任何StableDiffusion1.5的模型转化为一秒出图的bmodel

我们建议在docker内转模型，以保证环境一致性。

## 环境配置
```sh
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest`
```
进入镜像后  

```sh
docker exec -it myname bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:/aigc/tpu_mlir-1.6.404-py3-none-any.whl
pip3 install tpu_mlir-1.6.404-py3-none-any.whl
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

bash prepare.sh
```

### 在docker内执行命令

将模型从safetensor转为pt/onnx  
```sh
python3 export_from_safetensor.py -u xxxxx/model.safetensor -l xxxx/lora.safetensor -c xxxx/controlnet.safetensor -b 1 -o xxxxx/name 
```
这里需要考虑到这一点：如果没有controlnet可以不传-c参数，如果没有lora可以不传-l参数 
模型最后保存到-o指定的目录里面 
目录里面应该会变成：  
```
.
├── text_encoder
│   └── text_encoder.onnx
├── unet
│   └── unet_fuse_1.pt
├── vae_decoder
│   └── vae_decoder.pt
└── vae_encoder
    └── vae_encoder.pt
```
第二步将pt/onnx转为bmodel 
```sh 
python3 convert_bmodel.py -n xxxxx/name -o xxxxx -s 512 512 -b 1 -v sd15
```
得到的bmodel 在 `-o xxxxx` 的目录里面 
结果是这样：
```
.
├── sdv15_text.bmodel
├── sdv15_unet_multisize.bmodel
├── sdv15_vd_multisize.bmodel
└── sdv15_ve_multisize.bmodel
```

- `-s 512 512` 是image size, 必须是8的整数倍，相应的`latent shape`是`[1,4,size[0]//8,size[1]//8]`。如果想接入多个shape，则可以这么写 `-s w h w2 h2` 注意shape不能太大 不要超过1024 

## 3. 使用bmodel
在`model_path.py`中模型列表的最前面填上新模型的路径信息（应用默认加载该列表中的第一个模型）。

## 常见问题
1. 转换bmodel后，出图明显异常。

    Solution：使用较低版本的torch（如1.12.0）重新跑`model_export`中的trace脚本，再重新转bmodel。
2. bmodel出图结果make sense，但与纯CPU上跑出的结果不完全一致。
    
    Solution：检查text encoder输出差异是否过大。C站上SD1.5的不同checkpoint，Unet和text encoder都不同，都需要重新trace和转换bmodel！另外，可比对F32、F16、BF16的输出差异。

3. 如果报错text encoder， vae之类找不到，这是因为safetensor可能不包括text encoder或vae的权重。因为这种问题很少见，暂时不支持，如果遇到，请给提issue。 
