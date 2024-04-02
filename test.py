from sd import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import time
import random
import torch
# from sd.untool import delete_runtime, free_runtime
from model_path import model_path

DEVICE_ID = 0
# 对路径的要求：
# 模型必须放在SD-lcm-tpu/models/basic/文件夹下
#   比如你的模型名字为 aaaa 那么路径就是 SD-lcm-tpu/models/basic/aaaa 
BASENAME = list(model_path.keys())
SIZE = [("512:512", 512), ("768:768", 768)]
print(BASENAME)
scheduler = "LCM"

def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

seed_torch(1111)
class ModelManager():
    def __init__(self, model_name=None, size=512):
        if model_name in BASENAME:
            self.current_model_name = model_name
        else:
            assert len(BASENAME) > 0, "No model found"
            self.current_model_name = BASENAME[0]
        self.pipe = None
        self.size = size
        self.change_model(self.current_model_name, size=self.size)

    def pre_check(self, model_select, size, check_type=None):
        check_pass = True
        model_select_path = os.path.join('models', 'basic', model_select)
        te_path = os.path.join(model_select_path, model_path[model_select]['encoder'])
        unet_512_path = os.path.join(model_select_path, model_path[model_select]['unet']['512'])
        unet_768_path = os.path.join(model_select_path, model_path[model_select]['unet']['768'])
        vae_de_path = os.path.join(model_select_path, model_path[model_select]['vae_decoder'])
        vae_en_path = os.path.join(model_select_path, model_path[model_select]['vae_encoder'])

        if "te" in check_type:
            if not os.path.isfile(te_path):
                Warning("No {} please download first".format(model_select))
                check_pass = False
                # return False
        if "unet" in check_type:
            if size == 512:
                if not os.path.isfile(unet_512_path):
                    Warning("No {} unet_512 please download first".format(model_select))
                    check_pass = False
            else:
                if not os.path.isfile(unet_768_path):
                    Warning("No {} unet_768 please download first".format(model_select))
                    check_pass = False

        if "vae" in check_type:
            if not os.path.exists(vae_en_path) or not os.path.exists(vae_de_path):
                Warning("No {} vae please download first".format(model_select))
                check_pass = False

        return check_pass

    def change_model(self, model_select, size):
        if self.pipe is None:
            self.pre_check(model_select, size, check_type=["te", "unet", "vae"])
            self.pipe = StableDiffusionPipeline(
                basic_model=model_select,
                scheduler=scheduler,
            )
            self.pipe.set_height_width(size, size)
            self.current_model_name = model_select
            self.size = size
            return

        if self.current_model_name != model_select or self.size != size:
            if self.current_model_name != model_select:
                # change both te and unet
                if self.pre_check(model_select, size, check_type=["te", "unet"]):
                    try:
                        self.pipe.change_lora(model_select, size)
                        self.pipe.set_height_width(size, size)
                        self.current_model_name = model_select
                        self.size = size
                        return model_select, size
                    except Exception as e:
                        print(e)
                        return self.current_model_name, self.size
                else:
                    return self.current_model_name, self.size


            elif self.current_model_name == model_select and self.size != size:
                # only change the unet
                if self.pre_check(model_select, size, check_type=["unet"]):
                    try:
                        self.pipe.change_unet(model_select, size)
                        self.pipe.set_height_width(size, size)
                        self.size = size
                        return model_select, size
                    except Exception as e:
                        print(e)
                        return self.current_model_name, self.size
                else:
                    return self.current_model_name, self.size

        else:
            return self.current_model_name, self.size

    def generate_image_from_text(self, text, image=None, step=4, strength=0.5, seed=None, crop=None, controlnet_img=None, controlnet_weight=1.0,controlnet_args={}):
        img_pil = self.pipe(
            init_image=image,
            prompt=text,
            negative_prompt="low resolution",
            num_inference_steps=step,
            strength=strength,
            scheduler=scheduler,
            guidance_scale=0,
            controlnet_img=controlnet_img,
            seeds=[random.randint(0, 1000000) if seed is None else seed],
            controlnet_args=controlnet_args,
            controlnet_weight=controlnet_weight
        )
        if crop == 1:
            h, w = img_pil.size
            print(h, w)
            img_pil = img_pil.crop((1/8*w, 0, 7/8*w, h))
        return img_pil

name = "meinamix"
model_manager = ModelManager(name, 512)

# text2img
text = "a photo of cat"
img = model_manager.generate_image_from_text(text, step=4, strength=1)
img.save("test_kh.png")


text = "a photo of cat"
controlnet_img = Image.open("test_kh.png")
img = model_manager.generate_image_from_text(text, step=4, strength=1, controlnet_img=controlnet_img, controlnet_weight =0.1, controlnet_args={
    "low_threshold": 100,
    "height_threshold": 140,
    "save_canny": True,# will store canny image into "canny.jpg"
    "start":1,
    "end":2
})
# end can be -1 if you want to use the last step
img.save("test_kh2w0.png")


# img2img
text = "a photo of dog"
controlnet_img = Image.open("test_kh.png")
source_img = Image.open("test_kh.png")
img = model_manager.generate_image_from_text(text, step=10, strength=0.7, image=source_img, controlnet_img=controlnet_img, controlnet_weight =0.8, controlnet_args={
    "low_threshold": 150,
    "height_threshold": 250,
    "save_canny": True
})
img.save("test_kh3.png")

# img2img+controlnet
text = "a photo of dog"
controlnet_img = Image.open("test_kh.png")
source_img = Image.open("test_kh.png")
img = model_manager.generate_image_from_text(text, step=10, strength=0.7, image=source_img)
img.save("test_kh4.png")