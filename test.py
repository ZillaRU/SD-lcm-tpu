from sd import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import time
import random
import torch
# from sd.untool import delete_runtime, free_runtime
from model_path import model_path
from sd.scheduler import samplers_k_diffusion

DEVICE_ID = 0
# 对路径的要求：
# 模型必须放在SD-lcm-tpu/models/basic/文件夹下
#   比如你的模型名字为 aaaa 那么路径就是 SD-lcm-tpu/models/basic/aaaa 
BASENAME = list(model_path.keys())
def create_size(*size_elements):
    unique_size_elements = sorted(list(set(size_elements)))
    all_sizes = []
    for i in unique_size_elements:
        for j in unique_size_elements:
            all_sizes.append([i, j])
    return [ (f"{size[0]}:{size[1]}", size) for size in all_sizes]

SIZE = create_size(512, 768) # [('512:512', [512,512]), ] W, H
print(BASENAME)
# scheduler = "Euler a"

scheduler = ["LCM", "DDIM"]
for i in samplers_k_diffusion:
   scheduler.append(i[0])

# bad_scheduler = ["DPM Solver++", "DPM fast", "DPM adaptive"]
# for i in bad_scheduler:
#     scheduler.remove(i)

class gr:
    @classmethod
    def Info(cls, msg):
        print(msg)
    
    @classmethod
    def Warning(cls, msg):
        print(msg)

    @classmethod
    def Error(cls, msg):
        print(msg)

    @classmethod
    def Progress(cls, *args, **kwargs):
        def inner(*args, **kwargs):
            print(args, kwargs)
        return inner
    
    @classmethod
    def Slider(cls, *args, **kwargs):
        def inner(*args, **kwargs):
            print(args, kwargs)
        return inner

def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

seed_torch(1111)

class ModelManager():
    def __init__(self,name=None, scheduler=scheduler[0]):
        self.current_model_name = None
        self.pipe = None
        self.current_scheduler = scheduler
        if name:
            self.change_model(name, scheduler=scheduler)
        else:
            self.change_model(BASENAME[0], scheduler=scheduler)

    def pre_check_latent_size(self, latent_size):
        latent_size_str = "{}x{}".format(SIZE[latent_size][1][0], SIZE[latent_size][1][1])
        support_status = model_path[self.current_model_name]["latent_shape"][latent_size_str]
        if support_status == "True":
            return True
        else:
            return False

    def pre_check(self, model_select, check_type=None):
        return True

    def change_model(self, model_select, scheduler=None, progress=gr.Progress()):
        if self.pipe is None:
            # self.pre_check(model_select, check_type=["te", "unet", "vae"])
            self.pipe = StableDiffusionPipeline(
                basic_model=model_select,
                scheduler=scheduler
            )
            self.current_model_name = model_select
            return

        if self.current_model_name != model_select:
            # change both te, unet, vae
            if self.pre_check(model_select, check_type=["te", "unet", "vae"]):
                try:
                    gr.Info("Loading {} ...".format(model_select))
                    progress(0.4, desc="Loading....")
                    self.pipe.change_lora(model_select)
                    progress(0.8, desc="Loading....")
                    gr.Info("Success load {} LoRa".format(model_select))
                    self.current_model_name = model_select
                    return model_select
                except Exception as e:
                    print(e)
                    gr.Error("{}".format(e))
                    return self.current_model_name
            else:
                return self.current_model_name

        else:
            gr.Info("{} LoRa have been loaded".format(model_select))
            return self.current_model_name

    def generate_image_from_text(self, text, image=None, step=4, strength=0.5, seed=None, latent_size=None, scheduler=None, guidance_scale=None, enable_prompt_weight=None, negative_prompt=None,
                                 controlnet_img=None, controlnet_weight=1.0,controlnet_args={}):
        if self.pre_check_latent_size(latent_size):
            self.pipe.set_height_width(SIZE[latent_size][1][1], SIZE[latent_size][1][0])
            img_pil = self.pipe(
                init_image=image,
                prompt=text,
                negative_prompt=negative_prompt,
                num_inference_steps=step,
                strength=strength,
                scheduler=scheduler,
                guidance_scale=guidance_scale,
                enable_prompt_weight = enable_prompt_weight,
                seeds=[random.randint(0, 1000000) if seed is None else seed],
                controlnet_img=controlnet_img,
                controlnet_args=controlnet_args,
                controlnet_weight=controlnet_weight
            )

            return img_pil
        else:
            gr.Info("{} do not support this size, please check model info".format(self.current_model_name))

    def update_slider(self, scheduler):
        if scheduler != self.current_scheduler and scheduler == "LCM":
            self.current_scheduler = scheduler
            return gr.Slider(minimum=3, maximum=10, step=1, value=4, label="Steps", scale=2)
        elif scheduler != self.current_scheduler and self.current_scheduler == "LCM":
            self.current_scheduler = scheduler
            return gr.Slider(minimum=15, maximum=40, step=1, value=20, label="Steps", scale=2)
        else:
            return 20

model_name = "meinamix"
model_manager = ModelManager(model_name)

prompt = "a beautiful landscape painting"
negative_prompt = "low quality, bad resolution"
latent_size = 0
scheduler = "Euler a"
step = 4
guidance_scale = 0.9
img = model_manager.generate_image_from_text(prompt, step=step, negative_prompt=negative_prompt, latent_size=latent_size, scheduler=scheduler, guidance_scale=guidance_scale)
img.save("test.png")

controlnet_img = Image.open("test.png")
controlnet_weight = 1
controlnet_args={
    "low_threshold": 150,
    "height_threshold": 250,
    "save_canny": True
}
guidance_scale = 1.2
img = model_manager.generate_image_from_text(prompt, step=step, negative_prompt=negative_prompt, latent_size=latent_size, scheduler=scheduler, guidance_scale=guidance_scale, controlnet_img=controlnet_img, controlnet_weight=controlnet_weight, controlnet_args=controlnet_args)
img.save("test_controlnet.png")
