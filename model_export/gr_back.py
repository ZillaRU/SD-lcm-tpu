import gradio as gr
import time
from ConvertorPtOnnx import ConvertorPtOnnx
from ConvertorBmodel import ConvertorBmodel
import os

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

    def init_convertor_wapper(self):
        # gr.Info("Intting Convertor")
        self.convertor.init_convertor()
        return "Initialized Convertor success"

    def init_pipe_wapper(self):
        # gr.Info("Initting Pipe")
        self.convertor.init_pipe()
        return "Initialized Pipe success"

    def run_controlnet_wapper(self):
        gr.Info("Running Controlnet")
        self.convertor.run_controlnet()
        return "Run controlnet success"

    def run_unet_wapper(self):
        gr.Info("Running Unet")
        self.convertor.run_unet()
        return "Run Unet success"

    def run_text_encoder_wapper(self):
        gr.Info("Running Text Encoder")
        self.convertor.run_text_encoder()
        return "Run Text Encoder success"

    def run_vae_wapper(self):
        gr.Info("Running VAE")
        self.convertor.run_vae()
        return "Run VAE success"


def preprocess(unet_path, dk_unet_path, unet_url, controlnet_path, dk_controlnet_path, controlnet_url, lora_path, dk_lora_path, lora_url):
    check = True
    if unet_path is not None:
        gr.Info("Use Upload Unet safetensor")
    elif unet_path is None and dk_unet_path is not None:
        gr.Info("Use docker Unet safetensor")
        unet_path = dk_unet_path
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
             lora_url=None, batch=None, version=None, merge=None, output_name=None, debug=None, progress=gr.Progress()):
    # print(output_name)
    progress(0, desc="Starting...")
    # print(unet_path)
    # print(dk_unet_path)
    # print(unet_url, type(unet_url))

    # print(type(lora_path), len(lora_path))

    unet_path, controlnet_path, lora_path, check = preprocess(unet_path, dk_unet_path, unet_url, controlnet_path, dk_controlnet_path, controlnet_url, lora_path, dk_lora_path, lora_url)
    # print(lora_path)
    if not check:
        gr.Warning("Please Upload or Select a correct file")
        return "fuck your mm"
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
    fn_list = [gr_convertor.init_convertor_wapper,
               gr_convertor.init_pipe_wapper,
               gr_convertor.run_controlnet_wapper,
               gr_convertor.run_unet_wapper,
               gr_convertor.run_text_encoder_wapper,
               gr_convertor.run_vae_wapper]
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

    def convert_sd15_unet_wapper(self):
        gr.Info("Converting Unet to Bmodel")
        self.convertor.convert_sd15_unet()

    def convert_sd15_controlnet_wapper(self):
        # gr.Info()
        self.convertor.convert_sd15_controlnet()


    def convert_sd15_text_encoder_wapper(self):
        gr.Info("Converting Text Encoder to Bmodel")
        self.convertor.convert_sd15_text_encoder()

    def convert_sd15_vae_encoder_wapper(self):
        gr.Info("Converting VAE Encoder to Bmodel")
        self.convertor.convert_sd15_vae_encoder()

    def convert_sd15_vae_decoder_wapper(self):
        gr.Info("Converting VAE Decoder to Bmodel")
        self.convertor.convert_sd15_vae_decoder()

    def move_bmodels_into_folder_wapper(self):
        self.convertor.move_bmodels_into_folder()
        gr.Info("Convert Bmodels Finish")


def run_back_2(shape_h, shape_w, version, path, batch, output_bmodel="", progress=gr.Progress()):
    shape_lists = [[shape_h, shape_w]]
    progress(0, desc="Starting...")
    gr_convertor = GrConvertorBmodel(shape_lists, version, path, batch, output_bmodel)

    fn_list = [gr_convertor.convert_sd15_unet_wapper,
               gr_convertor.convert_sd15_controlnet_wapper,
               gr_convertor.convert_sd15_text_encoder_wapper,
               gr_convertor.convert_sd15_vae_encoder_wapper,
               gr_convertor.convert_sd15_vae_decoder_wapper,
               gr_convertor.move_bmodels_into_folder_wapper]

    try:
        for i in progress.tqdm(range(6)):
            fn_list[i]()

        return "Convert to Bmodels Success, please check {}".format(gr_convertor.convertor.output_bmodel)
    except Exception as e:
        print(e)
        gr.Warning("Error check the details in terminal")
        return "Convert to Bmodels Failed, Please check and retry"










