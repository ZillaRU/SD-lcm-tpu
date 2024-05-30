import gradio as gr
import time
from ConvertorPtOnnx import ConvertorPtOnnx
from ConvertorBmodel import ConvertorBmodel
import os
import requests
import uuid
import subprocess
import re

class GrConvertorPtOnnx():
    def __init__(self, unet_path, controlnet_path=None, lora_path=None, unet_url=None, controlnet_url=None, lora_url=None, batch=None, version=None, merge=None, output_name=None, debug=None):
        self.convertor = ConvertorPtOnnx(
            unet_path=unet_path,
            controlnet_path=controlnet_path,
            lora_path=lora_path,
            merge=merge,
            batch=batch,
            version=version,
            output_name=output_name,
            debug=debug
        )

    def init_convertor_wrapper(self):
        # gr.Info("Intting Convertor")
        self.convertor.init_convertor()
        return "Initialized Convertor success"

    def init_pipe_wrapper(self):
        # gr.Info("Initting Pipe")
        self.convertor.init_pipe()
        return "Initialized Pipe success"

    def run_controlnet_wrapper(self):
        gr.Info("Running Controlnet")
        self.convertor.run_controlnet()
        return "Run controlnet success"

    def run_unet_wrapper(self):
        gr.Info("Running Unet")
        self.convertor.run_unet()
        return "Run Unet success"

    def run_text_encoder_wrapper(self):
        gr.Info("Running Text Encoder")
        self.convertor.run_text_encoder()
        return "Run Text Encoder success"

    def run_vae_wrapper(self):
        gr.Info("Running VAE")
        self.convertor.run_vae()
        return "Run VAE success"


def civital_api_download(url, token):
    if not os.path.exists('./URL_civital_models'):
        os.makedirs('./URL_civital_models', exist_ok=True)
    if token == None:
        gr.Info("Please add your Civital API token")
    else:
        try:
            gr.Info("Downloading model ...")
            model_name = "{}.safetensors".format(uuid.uuid4())
            cmd = "wget -O ./URL_civital_models/{} \"{}?&token={}\"".format(model_name, url, token)
            print(cmd)
            ret = subprocess.run(cmd, shell=True, check=True)
            # print("Download {} successfully".format(original_model_name))
            return "./URL_civital_models/{}".format(model_name)
        except Exception as e:
            print(e)
            gr.Warning("Can not download the model, please check the Internet and URL")
            return None



def preprocess(unet_path, dk_unet_path, unet_url, controlnet_path, dk_controlnet_path, controlnet_url, lora_path, dk_lora_path, lora_url, token):
    check = True
    if unet_path is not None:
        gr.Info("Use Upload Unet safetensor")
    elif unet_path is None and dk_unet_path is not None:
        gr.Info("Use docker Unet safetensor")
        unet_path = dk_unet_path
    elif unet_path is None and dk_unet_path is None and unet_url is not None:
        gr.Info("Download the file via Internet")
        path = civital_api_download(unet_url, token)
        if path is not None:
            unet_path = path
        else:
            check = False
    else:
        gr.Warning("Please Upload or Select a Unet safetensor file")
        check = False

    if controlnet_path is not None:
        gr.Info("Use Upload controlnet")
    elif controlnet_path is None and dk_controlnet_path is not None:
        gr.Info("Use docker controlnet")
        controlnet_path = dk_controlnet_path[0]

    if lora_path is not None:
        gr.Info("Use Upload lora")
        lora_path = os.path.split(lora_path[0])[0]
        # lora_path = lora_path[0].split('/', -1)
    elif lora_url is None and dk_lora_path is not None:
        gr.Info("Use docker controlnet")
        lora_path = dk_lora_path[0]

    return unet_path, controlnet_path, lora_path, check



def run_back_1(unet_path, controlnet_path=None, lora_path=None, dk_unet_path=None, dk_controlnet_path=None, dk_lora_path=None, unet_url=None, controlnet_url=None,
             lora_url=None, batch=None, version=None, merge=None, output_name=None, debug=None, token=None, progress=gr.Progress()):
    # print(output_name)
    progress(0, desc="Starting...")
    # print(unet_path)
    # print(dk_unet_path)
    # print(unet_url, type(unet_url))

    # print(type(lora_path), len(lora_path))

    unet_path, controlnet_path, lora_path, check = preprocess(unet_path, dk_unet_path, unet_url, controlnet_path, dk_controlnet_path, controlnet_url, lora_path, dk_lora_path, lora_url, token)
    # print(lora_path)
    if not check:
        gr.Warning("Please Upload or Select a correct file")
        return "Please Upload or Select a correct file"
    # print(output_name)
    gr_convertor = GrConvertorPtOnnx(
        unet_path=unet_path,
        controlnet_path=controlnet_path,
        lora_path=lora_path,
        merge=merge,
        batch=batch,
        version=version,
        output_name=output_name,
        debug=debug
    )
    fn_list = [gr_convertor.init_convertor_wrapper,
               gr_convertor.init_pipe_wrapper,
               gr_convertor.run_controlnet_wrapper,
               gr_convertor.run_unet_wrapper,
               gr_convertor.run_text_encoder_wrapper,
               gr_convertor.run_vae_wrapper]
    try:
        for i in progress.tqdm(range(6)):
            # time.sleep(0.5)
            fn_list[i]()

        return "Convert to Pt/Onnx Success, please check {}".format(gr_convertor.convertor.output_name)
    except Exception as e:
        print(e)
        gr.Warning("Error check the details in terminal")
        return "Convert to Pt/Onnx Failed, Please check and retry"


class GrConvertorBmodel():
    def __init__(self, shape_lists, version, path, batch, output_bmodel):
        print(path)

        self.convertor = ConvertorBmodel(
            shape_lists=shape_lists,
            version=version,
            path=path,
            batch=batch,
            output_bmodel=output_bmodel
        )

    def convert_sd15_unet_wrapper(self):
        gr.Info("Converting Unet to Bmodel")
        self.convertor.convert_sd15_unet()

    def convert_sd15_controlnet_wrapper(self):
        # gr.Info()
        self.convertor.convert_sd15_controlnet()


    def convert_sd15_text_encoder_wrapper(self):
        gr.Info("Converting Text Encoder to Bmodel")
        self.convertor.convert_sd15_text_encoder()

    def convert_sd15_vae_encoder_wrapper(self):
        gr.Info("Converting VAE Encoder to Bmodel")
        self.convertor.convert_sd15_vae_encoder()

    def convert_sd15_vae_decoder_wrapper(self):
        gr.Info("Converting VAE Decoder to Bmodel")
        self.convertor.convert_sd15_vae_decoder()

    def move_bmodels_into_folder_wrapper(self):
        self.convertor.move_bmodels_into_folder()
        gr.Info("Convert Bmodels Finish")


def run_back_2(shape_lists_str, version, path, batch, output_bmodel="", progress=gr.Progress()):
    if path is None:
        gr.Warning("please select the model path folder")
        return
    def has_non_digit(s):
        pattern = r'[^\d\s]'
        return bool(re.search(pattern, s))

    if has_non_digit(shape_lists_str):
        gr.Warning("Only accept digit")
        return

    shape_nums = shape_lists_str.split(" ")

    shape_len = len(shape_nums)
    if shape_len % 2 != 0:
        gr.Warning("Please input valid shape lists, should be even")
        return
    else:
        shape_lists = []
        for i in range(0, shape_len, 2):
            shape_lists.append([int(shape_nums[i]), int(shape_nums[i+1])])

    progress(0, desc="Starting...")
    gr_convertor = GrConvertorBmodel(shape_lists, version, path, batch, output_bmodel)

    fn_list = [gr_convertor.convert_sd15_unet_wrapper,
               gr_convertor.convert_sd15_controlnet_wrapper,
               gr_convertor.convert_sd15_text_encoder_wrapper,
               gr_convertor.convert_sd15_vae_encoder_wrapper,
               gr_convertor.convert_sd15_vae_decoder_wrapper,
               gr_convertor.move_bmodels_into_folder_wrapper]

    try:
        for i in progress.tqdm(range(6)):
            fn_list[i]()

        return "Convert to Bmodels Success, please check {}".format(gr_convertor.convertor.output_bmodel)
    except Exception as e:
        print(e)
        gr.Warning("Error check the details in terminal")
        return "Convert to Bmodels Failed, Please check and retry"










