import gradio as gr
from sd import StableDiffusionPipeline
import os
import time
import random
from sd.scheduler import samplers_k_diffusion
import warnings
from PIL import Image
from utils.tools import create_size, ratio_resize, seed_torch, get_model_input_info, get_model_path

warnings.filterwarnings("ignore")
model_path = get_model_path()

DEVICE_ID = 0
BASENAME = list(model_path.keys())
CONTROLNET = os.listdir('./models/controlnet')

if len(CONTROLNET) != 0:
    CONTROLNET = [i.split('.')[0] for i in CONTROLNET]

scheduler = ["LCM", "DDIM", "DPM Solver++"]
for i in samplers_k_diffusion:
   scheduler.append(i[0])

bad_scheduler = ["DPM Solver++", "DPM fast", "DPM adaptive"]
for i in bad_scheduler:
    scheduler.remove(i)

SIZE = create_size(512, 768) # [('512:512', [512,512]), ] W, H


class ModelManager():
    def __init__(self):
        self.current_model_name = None
        self.pipe = None
        self.current_scheduler = scheduler[0]
        self.controlnet = None
        self.current_model_input_shapes = None


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

    def change_model(self, model_select, scheduler=None, controlnet=None, progress=gr.Progress()):
        if controlnet == []:
            controlnet = None
        if model_select == []:
            model_select = None
        if model_select is not None:
            if self.pipe is None:
                if self.pre_check(model_select, check_type=["te", "unet", "vae"]):
                    self.pipe = StableDiffusionPipeline(
                        basic_model=model_select,
                        scheduler=scheduler,
                        controlnet_name=controlnet,
                    )
                    self.current_model_name = model_select
                    self.controlnet = controlnet
                    self.current_model_input_shapes = get_model_input_info(self.pipe.unet.basic_info["stage_info"]) # W H

                return self.current_model_name, self.controlnet

            if self.current_model_name != model_select:
                # change both te, unet, vae
                if self.pre_check(model_select, check_type=["te", "unet", "vae"]):
                    try:
                        gr.Info("Loading {} ...".format(model_select))
                        progress(0.4, desc="Loading....")
                        self.pipe.change_lora(model_select)
                        progress(0.8, desc="Loading....")
                        gr.Info("Success load {} LoRa".format(model_select))
                        progress(0.9, desc="Loading....")
                        self.pipe.change_controlnet(controlnet)
                        self.current_model_name = model_select
                        self.controlnet = controlnet
                        self.current_model_input_shapes = get_model_input_info(self.pipe.unet.basic_info["stage_info"])  # W H


                    except Exception as e:
                        print(e)
                        gr.Error("{}".format(e))
                        return self.current_model_name, self.controlnet

                else:
                    return self.current_model_name, self.controlnet

            if self.controlnet != controlnet:
                try:
                    progress(0.9, desc="Loading....")
                    self.pipe.change_controlnet(controlnet)
                    self.controlnet = controlnet
                except Exception as e:
                    print(e)
                    gr.Error("{}".format(e))
                    return self.current_model_name, self.controlnet

            else:
                gr.Info("{} LoRa with {} have been loaded".format(model_select, controlnet))
                return self.current_model_name, self.controlnet

            return self.current_model_name, self.controlnet

        else:
            gr.Info("Please select a model")
            return None, None


    def generate_image_from_text(self, text, image=None, step=4, strength=0.5, seed=None, latent_size_index=None, scheduler=None, guidance_scale=None, enable_prompt_weight=None, negative_prompt=None, local_img=None):
        if image is None and local_img is not None:
            image = Image.open(local_img)
        if image is not None:
            target_size = SIZE[latent_size_index][1]
            image = ratio_resize(image, target_size)
        if self.pipe is None:
            gr.Info("Please select a model")
            return None
        elif SIZE[latent_size_index][1] in self.current_model_input_shapes:
            self.pipe.set_height_width(SIZE[latent_size_index][1][1], SIZE[latent_size_index][1][0])
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
            gr.Warning("{} do not support this size, please check model info".format(self.current_model_name))

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
                with gr.Tab("Upload"):
                    upload_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='pil', label="image")
                with gr.Tab("Airbox"):
                    local_img = gr.FileExplorer(glob='*.jpg', root_dir='./', label="image", file_count='single')

                with gr.Row():
                    num_step = gr.Slider(minimum=3, maximum=10, value=4, step=1, label="Steps", scale=2)
                    denoise = gr.Slider(minimum=0.2, maximum=1.0, value=0.5, step=0.1, label="Denoising Strength",
                                        scale=1)
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=0, maximum=20, value=0, step=0.1, label="CFG scale", scale=2)
                    enable_prompt_weight = gr.Checkbox(label="Prompt weight")

                with gr.Row():
                    seed_number = gr.Number(value=1, label="Seed", scale=1)
                    latent_size_index = gr.Dropdown(choices=[i[0] for i in SIZE], label="Size (W:H)", value=[i[0] for i in SIZE][0], type="index", interactive=True,scale=1)
                    scheduler_type = gr.Dropdown(choices=scheduler, value=scheduler[0], label="Scheduler", interactive=True,scale=1)
                with gr.Row():
                    clear_bt = gr.ClearButton(value="Clear",
                                              components=[input_content, upload_image, seed_number, denoise,
                                                          num_step, enable_prompt_weight, guidance_scale])
                    submit_bt = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                with gr.Row():
                    model_select = gr.Dropdown(choices=BASENAME, value=None, label="Model", interactive=True)
                    controlnet = gr.Dropdown(choices=CONTROLNET, value=None, label="Controlnet", interactive=True)
                    load_bt = gr.Button(value="Load Model", interactive=True)
                out_img = gr.Image(label="Output")

        with gr.Row():
            with gr.Column():
                example = gr.Examples(
                    label="Example",
                    examples=[
                              ["1girl, ponytail ,white hair, purple eyes, medium breasts, collarbone, flowers and petals, landscape, background, rose, abstract",
                               "ugly, poor details, bad anatomy",
                               0.5,
                               0,
                               "Euler a",
                               "512:768"],
                              ["upper body photo, fashion photography of cute Hatsune Miku, very long turquoise pigtails and a school uniform-like outfit. She has teal eyes and very long pigtails held with black and red square-shaped ribbons that have become a signature of her design, moonlight passing through hair.",
                               "ugly, poor details, bad anatomy",
                               0.5,
                               0.3,
                               "DPM++ 2M SDE Karras",
                               "512:768"],
                               ["a young woman stands at the center, extending her arms wide against a vast, overcast seascape. She is positioned on a stony beach, where dark, smooth pebbles cover the ground. The ocean is calm with gentle waves lapping at the shore. Her attire is stylishly casual, with a street fashion sports a fitted grey crop top and voluminous black cargo pants, paired with a studded black leather jacket that adds a touch of rebellious flair. A black beanie caps her long hair that falls partially across her face, and she holds a black designer tote bag in her outstretched hand. Her posture exudes a sense of freedom and joy, embodying a spontaneous moment captured against the moody backdrop of an overcast sky and the tranquil sea.",
                               "ugly, poor details, bad anatomy",
                               0.5,
                               0.2,
                               "LCM",
                               "512:768"],
                              ],
                    inputs=[input_content, negative_prompt, denoise, guidance_scale, scheduler_type, latent_size_index]
                )

        scheduler_type.change(model_manager.update_slider, scheduler_type, num_step)
        clear_bt.add(components=[out_img])
        load_bt.click(model_manager.change_model, [model_select, scheduler_type, controlnet], [model_select, controlnet])
        input_content.submit(model_manager.generate_image_from_text,
                             [input_content, upload_image, num_step, denoise, seed_number, latent_size_index, scheduler_type, guidance_scale, enable_prompt_weight, negative_prompt, local_img], [out_img])
        negative_prompt.submit(model_manager.generate_image_from_text,
                             [input_content, upload_image, num_step, denoise, seed_number, latent_size_index, scheduler_type, guidance_scale, enable_prompt_weight, negative_prompt, local_img], [out_img])
        submit_bt.click(model_manager.generate_image_from_text,
                        [input_content, upload_image, num_step, denoise, seed_number, latent_size_index, scheduler_type, guidance_scale, enable_prompt_weight, negative_prompt, local_img], [out_img])

    # 运行 Gradio 应用
    demo.queue(max_size=10)
    demo.launch(server_port=8999, server_name="0.0.0.0")
