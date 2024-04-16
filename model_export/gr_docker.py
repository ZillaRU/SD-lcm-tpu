import os
import time

from gr_back import *

description = """
# Stable Diffusion Model Convertor 🐥

## Two Steps:
- convert safetensor into pt/onnx. 
- convert pt/onnx into bmodel!

You can choose upload models or download by URL
"""
import gradio as gr

gr_convertor = None

if not os.path.exists('./tmp'):
    os.makedirs('tmp', exist_ok=True)


def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


# if not os.path.exists('./url_model'):
#     os.makedirs('url_model', exist_ok=True)

# controller = ConvertorController()




if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Tab("Convert to pt/onnx"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Tab("Upload"):
                            unet_safetensors_path = gr.File(file_types=['safetensors'], type='filepath',
                                                        label='unet safetensors')

                        with gr.Tab("Docker"):
                            docker_unet_safetensors_path = gr.FileExplorer(root_dir='./', label="unet safetensors path")

                        with gr.Tab("URL"):
                            unet_safetensors_url = gr.Textbox(label="Unet Safetensors URL", interactive=False)
                    with gr.Row():
                        with gr.Tab("Upload"):
                            controlnet_path = gr.File(file_types=['safetensors'], type='filepath', label='controlnet model')

                        with gr.Tab("Docker"):
                            docker_controlnet_path = gr.FileExplorer(root_dir='./', label="controlnet path")

                        with gr.Tab("URL"):
                            controlnet_url = gr.Textbox(label="Controlnet URL", interactive=False)

                    with gr.Row():
                        with gr.Tab("Upload"):
                            lora_path = gr.File(file_count="directory", type='filepath', label='lora model')

                        with gr.Tab("Docker"):
                            docker_lora_path = gr.FileExplorer(root_dir='./', label="lora path")

                        with gr.Tab("URL"):
                            lora_url = gr.Textbox(label="Lora URL", interactive=False)

                with gr.Column():
                    with gr.Tab("Config"):
                        # unet_safetensors_url = gr.Textbox(label="Unet Safetensors URL", interactive=False)
                        # controlnet_url = gr.Textbox(label="Controlnet URL", interactive=False)
                        # lora_url = gr.Textbox(label="Lora URL", interactive=False)
                        with gr.Row():
                            batch_num = gr.Number(value=1, label="batch", min_width=20)
                            version = gr.Dropdown(choices=['sd15', 'sd21', 'SDXL'], type='value', label="version",
                                                  value='sd15', interactive=True, min_width=160)
                            with gr.Column():
                                controlnet_merge_bool = gr.Checkbox(value=False, label='controlnet merge', min_width=20,
                                                                    info="merge unet into controlnet with the former")
                                debug_bool = gr.Checkbox(value=False, label='debug', min_width=20)
                        info_window = gr.Textbox(label="Progress info", lines=11)
                        clear_all_bt = gr.ClearButton(value="Clear All",
                                                      components=[unet_safetensors_path,
                                                                  controlnet_path,
                                                                  lora_path,
                                                                  unet_safetensors_url,
                                                                  controlnet_url,
                                                                  lora_url,
                                                                  controlnet_merge_bool,
                                                                  debug_bool,
                                                                  info_window])
                        step_1_bt = gr.Button(value="Convert to pt/onnx", variant='primary')
                        output_name = gr.Textbox(value=f"./tmp/{get_time_str()}", visible=False)

        with gr.Tab("Convert to bmodel"):
            with gr.Row():
                with gr.Column():
                    model_path = gr.FileExplorer(root_dir='./tmp', label="model path")
                    b_info_window = gr.Textbox(label="Progress info", lines=7)

                with gr.Column():
                    with gr.Row():
                        shape_h = gr.Number(value=512, label="shape_h", interactive=True)
                        shape_w = gr.Number(value=512, label="shape_w", interactive=True)
                    with gr.Row():
                        b_batch_num = gr.Number(value=1, label="batch", min_width=20)
                        b_version = gr.Dropdown(choices=['sd15', 'sd21', 'SDXL'], type='value', label="version",
                                                value='sd15',
                                                interactive=True, min_width=160)
                    b_clear_all_bt = gr.ClearButton(value="Clear All",
                                                    components=[shape_h,
                                                                shape_w,
                                                                info_window])
                    step_2_bt = gr.Button(value="Convert to bmodel", variant='primary')

        step_1_bt.click(run_back, [unet_safetensors_path,
                                   controlnet_path,
                                   lora_path,
                                   docker_unet_safetensors_path,
                                   docker_controlnet_path,
                                   docker_lora_path,
                                   unet_safetensors_url,
                                   controlnet_url,
                                   lora_url,
                                   batch_num,
                                   version,
                                   controlnet_merge_bool,
                                   output_name,
                                   debug_bool,
                                   info_window], [info_window])
        # step_1_bt.click(change_text, [info_window], )
        step_2_bt.click(convert_bmodel_wapper, [shape_h, shape_w, b_version, model_path, b_batch_num], [info_window])

    demo.queue(max_size=2)
    demo.launch(debug=False, show_api=True, share=False, server_name="0.0.0.0")
