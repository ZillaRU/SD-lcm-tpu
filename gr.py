import gradio as gr
from sd import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import time
import random
import torch
from model_path import model_path
from sd.scheduler import samplers_k_diffusion
from itertools import permutations

DEVICE_ID = 0
BASENAME = list(model_path.keys())

scheduler = ["LCM", "DDIM", "DPM Solver++"]
for i in samplers_k_diffusion:
   scheduler.append(i[0])

bad_scheduler = ["DPM Solver++", "DPM fast", "DPM adaptive"]
for i in bad_scheduler:
    scheduler.remove(i)

def create_size(*size_elements):
    unique_size_elements = set(size_elements)
    rectangle = list(permutations(unique_size_elements, 2))
    square = [(img_size, img_size) for img_size in unique_size_elements]
    all_img_size = square + rectangle
    return [(f"{size[0]}:{size[1]}", [size[0], size[1]]) for size in all_img_size]

SIZE = create_size(512, 768) # [('512:512', [512,512]), ] W, H


def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)


class ModelManager():
    def __init__(self):
        self.current_model_name = None
        self.pipe = None
        self.current_scheduler = scheduler[0]
        self.change_model(BASENAME[0], scheduler=scheduler[0])

    def pre_check_latent_size(self, latent_size):
        latent_size_str = "{}x{}".format(SIZE[latent_size][1][0], SIZE[latent_size][1][1])
        support_status = model_path[self.current_model_name]["latent_shape"].get(latent_size_str)
        if support_status == "True":
            return True
        else:
            return False


    def pre_check(self, model_select, check_type=None):
        check_pass = True
        model_select_path = os.path.join('models', 'basic', model_select)
        te_path = os.path.join(model_select_path, model_path[model_select]['encoder'])
        unet_path = os.path.join(model_select_path, model_path[model_select]['unet'])
        vae_de_path = os.path.join(model_select_path, model_path[model_select]['vae_decoder'])
        vae_en_path = os.path.join(model_select_path, model_path[model_select]['vae_encoder'])

        if "te" in check_type:
            if not os.path.isfile(te_path):
                gr.Warning("No {} text encoder, please download first".format(model_select))
                check_pass = False
                # return False
        if "unet" in check_type:
            if not os.path.isfile(unet_path):
                gr.Warning("No {} unet, please download first".format(model_select))
                check_pass = False

        if "vae" in check_type:
            if not os.path.exists(vae_en_path) or not os.path.exists(vae_de_path):
                gr.Warning("No {} vae, please download first".format(model_select))
                check_pass = False

        return check_pass

    def change_model(self, model_select, scheduler=None, progress=gr.Progress()):
        if self.pipe is None:
            self.pre_check(model_select, check_type=["te", "unet", "vae"])
            self.pipe = StableDiffusionPipeline(
                basic_model=model_select,
                scheduler=scheduler,
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

    def generate_image_from_text(self, text, image=None, step=4, strength=0.5, seed=None, latent_size=None, scheduler=None, guidance_scale=None, enable_prompt_weight=None, negative_prompt=None):
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
                seeds=[random.randint(0, 1000000) if seed is None else seed]
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


model_manager = ModelManager()


description = """
# Text-to-Image and Image-to-Image Generator

Generate images that incorporate both text descriptions and uploaded images, allowing you to create unique visual content.
"""

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                input_content = gr.Textbox(lines=1, label="Prompt")
                negative_prompt = gr.Textbox(lines=1, label="Negative prompt")
                upload_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='pil', label="image")
                with gr.Row():
                    num_step = gr.Slider(minimum=3, maximum=10, value=4, step=1, label="Steps", scale=2)
                    denoise = gr.Slider(minimum=0.2, maximum=1.0, value=0.5, step=0.1, label="Denoising Strength",
                                        scale=1)
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=0, maximum=20, value=0, step=0.5, label="CFG scale", scale=2)
                    enable_prompt_weight = gr.Checkbox(label="Prompt weight")

                with gr.Row():
                    seed_number = gr.Number(value=1, label="Seed", scale=1)
                    latent_size = gr.Dropdown(choices=[i[0] for i in SIZE], label="Size", value=[i[0] for i in SIZE][0], type="index", interactive=True,scale=1)
                    scheduler_type = gr.Dropdown(choices=scheduler, value=scheduler[0], label="Scheduler", interactive=True,scale=1)
                with gr.Row():
                    clear_bt = gr.ClearButton(value="Clear",
                                              components=[input_content, upload_image, seed_number, denoise,
                                                          num_step, enable_prompt_weight, guidance_scale])
                    submit_bt = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                with gr.Row():
                    model_select = gr.Dropdown(choices=BASENAME, value=BASENAME[0], label="Model", interactive=True)
                    change_bt = gr.Button(value="Change", interactive=True)
                out_img = gr.Image(label="Output")

        scheduler_type.change(model_manager.update_slider, scheduler_type, num_step)
        clear_bt.add(components=[out_img])
        change_bt.click(model_manager.change_model, [model_select, scheduler_type], [model_select])
        input_content.submit(model_manager.generate_image_from_text,
                             [input_content, upload_image, num_step, denoise, seed_number, latent_size, scheduler_type, guidance_scale, enable_prompt_weight, negative_prompt], [out_img])
        negative_prompt.submit(model_manager.generate_image_from_text,
                             [input_content, upload_image, num_step, denoise, seed_number, latent_size, scheduler_type, guidance_scale, enable_prompt_weight, negative_prompt], [out_img])
        submit_bt.click(model_manager.generate_image_from_text,
                        [input_content, upload_image, num_step, denoise, seed_number, latent_size, scheduler_type, guidance_scale, enable_prompt_weight, negative_prompt], [out_img])

    # 运行 Gradio 应用
    demo.queue(max_size=10)
    demo.launch(server_port=8999, server_name="0.0.0.0")
