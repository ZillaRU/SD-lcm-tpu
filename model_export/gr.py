import gradio as gr
import time
from gr_back import *
description = """
# Stable Diffusion Model Convertor 🐥

## Two Steps:
- convert safetensor into pt/onnx. 
- convert pt/onnx into bmodel!

You can choose upload models or download by URL
"""


def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Tab("Convert to pt/onnx"):
            with gr.Row():
                with gr.Column():
                    unet_safetensors_path = gr.File(file_types=['safetensors'], type='filepath', label='unet safetensors')
                    controlnet_path = gr.File(file_types=['safetensors'], type='filepath', label='controlnet model')
                    lora_path = gr.File(file_types=['safetensors'], type='filepath', label='lora model')

                with gr.Column():
                    unet_safetensors_url = gr.Textbox(label="Unet Safetensors URL")
                    controlnet_url = gr.Textbox(label="Controlnet URL")
                    lora_url = gr.Textbox(label="Lora URL")
                    with gr.Row():
                        batch_num = gr.Number(value=1, label="batch", min_width=20)
                        version = gr.Dropdown(choices=['sd15', 'sd21','SDXL'], type='value', label="version", value='sd15', interactive=True, min_width=160)
                        with gr.Column():
                            controlnet_merge_bool = gr.Checkbox(value=False, label='controlnet merge', min_width=20, info="merge unet into controlnet with the former")
                            debug_bool = gr.Checkbox(value=False, label='debug', min_width=20)
                    info_window = gr.Textbox(label="Progress info", lines=6)
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

        step_1_bt.click(export_from_safetensor_wapper, [unet_safetensors_path,
                                                        controlnet_path,
                                                        lora_path,
                                                        unet_safetensors_url,
                                                        controlnet_url,
                                                        lora_url,
                                                        batch_num,
                                                        version,
                                                        controlnet_merge_bool,
                                                        debug_bool], [info_window])

        with gr.Tab("Convert to bmodel"):
            with gr.Row():
                with gr.Column():
                    model_path = gr.FileExplorer(root_dir='./tmp', label="model path")
                    info_window = gr.Textbox(label="Progress info", lines=7)

                with gr.Column():
                    with gr.Row():
                        shape_h = gr.Number(value=512, label="shape_h", interactive=True)
                        shape_w = gr.Number(value=512, label="shape_w", interactive=True)
                    with gr.Row():
                        b_batch_num = gr.Number(value=1, label="batch", min_width=20)
                        b_version = gr.Dropdown(choices=['sd15', 'sd21', 'SDXL'], type='value', label="version", value='sd15',
                                                interactive=True, min_width=160)
                    b_clear_all_bt = gr.ClearButton(value="Clear All",
                                                  components=[shape_h,
                                                              shape_w,
                                                              info_window])
                    step_2_bt = gr.Button(value="Convert to bmodel", variant='primary')

        step_2_bt.click(convert_bmodel_wapper, [model_path, shape_h, shape_w, b_batch_num, b_version])

    demo.queue(max_size=2)
    demo.launch(debug=False, show_api=True, share=False, server_name="0.0.0.0")
