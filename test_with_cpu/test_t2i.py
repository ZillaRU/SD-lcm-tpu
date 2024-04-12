import os
import os.path as path
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import argparse
import random
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
parser = argparse.ArgumentParser()

def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

seed_torch(1111)
parser.add_argument('-u', '--unet_safetensors_path', type=str, default= None)
parser.add_argument('-c', '--controlnet_path',       type=str, default= None) # should be safetensors 
parser.add_argument('-l', '--lora_path',             type=str, default= None)
unet_size = 512
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_single_file(args.unet_safetensors_path, load_safety_checker=False)
pipe.load_lora_weights(args.lora_path)
pipe.fuse_lora()
prompt = "a photo of cat"
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
img = pipe(prompt, height=512, width=512, num_inference_steps=4, guidance_scale=0).images[0]
img.save("output_cpu.png")