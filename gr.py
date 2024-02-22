import gradio as gr
from sd import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import time
import random
import torch

def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

DEVICE_ID = 0
BASENAME = "awportrait"
scheduler = "LCM"

pipe = StableDiffusionPipeline(
    basic_model=BASENAME,
    scheduler=scheduler
)
pipe.set_height_width(512, 512)

def generate_image_from_text(text, image=None, step=4, strength=0.5, seed=None):
    img_pil = pipe(
        init_image=image,
        prompt=text,
        negative_prompt="low resolution",
        num_inference_steps=step,
        strength=strength,
        scheduler=scheduler,
        guidance_scale=0,
        seeds=[random.randint(0, 1000000) if seed is None else seed]
    )
    return img_pil

num_step = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="#Steps")
slider = gr.Slider(minimum=0.5, maximum=1.0, value=0.5, step=0.1, label="Denoising Strength")

# 创建 Gradio 接口
iface = gr.Interface(
    fn=generate_image_from_text,  # 指定处理函数
    inputs=["text", "image", num_step, slider, "number"],  # 输入类型为文本和图像
    outputs="image",  # 输出类型为图像
    title="Text-to-Image and Image-to-Image Generator",  # 界面标题
    description="Generate images that incorporate both text descriptions and uploaded images, allowing you to create unique visual content."
)

# 运行 Gradio 应用
iface.launch(server_port=8999, server_name="0.0.0.0")