import gradio as gr
from sd import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import time
import random
import torch
# from sd.untool import delete_runtime, free_runtime
from mode_path import model_path
DEVICE_ID = 0
BASENAME = ["awportrait", "babes20lcm"]
scheduler = "LCM"

def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

class ModelManager():
    def __init__(self):
        self.init_status = False
        self.current_model_name = BASENAME[0]
        self.pipe = None
        self.model_path = model_path
        self.change_model(self.current_model_name)


    def change_model(self, model_select):
        if self.current_model_name != model_select or not self.init_status:
            self.current_model_name = model_select
            del self.pipe
            self.pipe = None
            self.pipe = StableDiffusionPipeline(
                basic_model=model_select,
                scheduler=scheduler,
                model_path_dict=self.model_path
            )
            self.pipe.set_height_width(512, 512)
            self.init_status = True


    def generate_image_from_text(self, text, image=None, step=4, strength=0.5, seed=None):
        img_pil = self.pipe(
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
                input_content = gr.Textbox(lines=1, label="Input content")
                upload_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='pil', label="image")
                num_step = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="Steps")
                denoise = gr.Slider(minimum=0.5, maximum=1.0, value=0.5, step=0.1, label="Denoising Strength")
                seed_number = gr.Number(value=1, label="seed")
                with gr.Row():
                    clear_bt = gr.ClearButton(value="Clear",
                                              components=[input_content, upload_image, seed_number, denoise,
                                                          num_step])
                    submit_bt = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                with gr.Row():
                    model_select = gr.Dropdown(choices=BASENAME, value=BASENAME[0], label="Model", interactive=False)
                    change_bt = gr.Button(value="Change", interactive=False)
                out_img = gr.Image(label="Output")

        clear_bt.add(components=[out_img])
        change_bt.click(model_manager.change_model, [model_select])
        input_content.submit(model_manager.generate_image_from_text, [input_content, upload_image, num_step, denoise, seed_number], [out_img])
        submit_bt.click(model_manager.generate_image_from_text, [input_content, upload_image, num_step, denoise, seed_number], [out_img])

    # 运行 Gradio 应用
    demo.queue()
    demo.launch(server_port=8999, server_name="0.0.0.0")