# current only support sd15 
import warnings
warnings.filterwarnings("ignore")
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import numpy as np
from safetensors.torch import load_file
from safetensors.torch import save_file
import argparse
from diffusers import StableDiffusionPipeline
import time
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)  

def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--unet_safetensors_path', type=str, default= None)
parser.add_argument('-c', '--controlnet_path',       type=str, default= None) # should be safetensors
parser.add_argument('-l', '--lora_path',             type=str, default= None)
parser.add_argument('-cm', '--controlnet_merge',     type=bool, default= False, help="merge unet into controlnet with the former: \n "+
                                                            "new_controlnet = controlnet_weight - sd_base_encoder_weight + cur_unet_encoder_weight")
parser.add_argument("-b", "--batch",                 type=int, default=1)
parser.add_argument("-v", "--version",               type=str, default="sd15")
parser.add_argument('-o', '--output_name',           type=str, default= f"./tmp/{get_time_str()}") # output_name should starts with ./tmp/
parser.add_argument('-debug', '--debug_log',         type=bool, default= False)
unet_size = 512
args = parser.parse_args()
if args.debug_log:
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    logging.getLogger("diffusers").setLevel(logging.DEBUG)
assert args.version in ["sd15"] , "only support sd15"
assert args.batch in [1,2], "only support batch 1 or 2"
print(args)
args.output_name = args.output_name + "_" + args.version

os.makedirs(args.output_name, exist_ok=True)
# 有lora必须有safetensors path
class emptyclass:
    pass
pipe = emptyclass()
if args.unet_safetensors_path is not None:
    pipe = StableDiffusionPipeline.from_single_file(args.unet_safetensors_path, load_safety_checker=False)
    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path)
        pipe.fuse_lora()
    for para in pipe.unet.parameters():
        para.requires_grad = False
    for para in pipe.text_encoder.parameters():
        para.requires_grad = False
    for para in pipe.vae.parameters():
        para.requires_grad = False

if args.controlnet_path is not None:
    original_config_file = "./safetensors/controlnet/config.yaml"
    pipe.controlnet = download_controlnet_from_original_ckpt(
        checkpoint_path=args.controlnet_path,
        original_config_file=original_config_file,
        from_safetensors=True,
        device="cpu"
    )
    for para in pipe.controlnet.parameters():
        para.requires_grad = False

def myunet(
    sample,
    timestep,
    encoder_hidden_states,
    mid_block_additional_residual,
    down_block_additional_residuals_0,
    down_block_additional_residuals_1,
    down_block_additional_residuals_2,
    down_block_additional_residuals_3,
    down_block_additional_residuals_4,
    down_block_additional_residuals_5,
    down_block_additional_residuals_6,
    down_block_additional_residuals_7,
    down_block_additional_residuals_8,
    down_block_additional_residuals_9,
    down_block_additional_residuals_10,
    down_block_additional_residuals_11,
    ):
    down_block_additional_residuals = [down_block_additional_residuals_0, down_block_additional_residuals_1, down_block_additional_residuals_2, down_block_additional_residuals_3, down_block_additional_residuals_4, down_block_additional_residuals_5, down_block_additional_residuals_6, down_block_additional_residuals_7, down_block_additional_residuals_8, down_block_additional_residuals_9, down_block_additional_residuals_10, down_block_additional_residuals_11]
    ret = pipe.unet(
        sample,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        cross_attention_kwargs=None,
        down_block_additional_residuals=down_block_additional_residuals,
        mid_block_additional_residual=mid_block_additional_residual,
    )
    return ret.sample

def create_controlnet_input():
    batch = args.batch
    img_size = (512, 512)
    controlnet_latent_model_input = torch.rand(batch, 4, img_size[0]//8, img_size[1]//8)
    controlnet_prompt_embeds = torch.rand(batch, 77, 768)
    image = torch.rand(batch, 3, img_size[0], img_size[1])
    t = torch.tensor([999])
    w = torch.tensor([1])
    return controlnet_latent_model_input, controlnet_prompt_embeds, image, t, w

def mycontrolnet(controlnet_latent_model_input, controlnet_prompt_embeds, image, t, weight):
    ret1, ret2= pipe.controlnet(
        controlnet_latent_model_input,
        t,
        controlnet_prompt_embeds,
        controlnet_cond = image,
        conditioning_scale=weight,
        guess_mode=False,
        return_dict=False,
    )
    return ret1[0], ret1[1], ret1[2], ret1[3], ret1[4], ret1[5],  ret1[6], ret1[7], ret1[8], ret1[9], ret1[10], ret1[11],  ret2

def create_unet_input():
    img_size = (unet_size, unet_size)
    batch = args.batch
    latent_model_input = torch.rand(batch, 4, img_size[0]//8, img_size[1]//8)
    t = torch.tensor([999])
    prompt_embeds = torch.rand(batch, 77, 768)
    mid_block_additional_residual = torch.rand(batch, 1280, img_size[0]//64, img_size[1]//64)
    down_block_additional_residuals = []
    down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0]//8, img_size[1]//8))
    down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0]//8, img_size[1]//8))
    down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0]//8, img_size[1]//8))
    down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0]//16, img_size[1]//16))
    down_block_additional_residuals.append(torch.rand(batch, 640, img_size[0]//16, img_size[1]//16))
    down_block_additional_residuals.append(torch.rand(batch, 640, img_size[0]//16, img_size[1]//16))
    down_block_additional_residuals.append(torch.rand(batch, 640, img_size[0]//32, img_size[1]//32))
    down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0]//32, img_size[1]//32))
    down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0]//32, img_size[1]//32))
    down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0]//64, img_size[1]//64))
    down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0]//64, img_size[1]//64))
    down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0]//64, img_size[1]//64))

    return (latent_model_input, t, prompt_embeds, mid_block_additional_residual, *down_block_additional_residuals)

def myte(te_input):
    ret = pipe.text_encoder(te_input)
    return ret.last_hidden_state

def create_te_input():
    return torch.Tensor([[0]*77]).long()

def export_vaencoder():
    log.info(f" start vae encoder convert")
    vae = pipe.vae
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = vae.encoder
            self.encoder2 = vae.quant_conv
        
        def forward(self, x):
            x = self.encoder1(x)
            x = self.encoder2(x)
            return x
    
    encoder = Encoder()
    img_size = (512, 512)
    img_size = (img_size[0]//8, img_size[1]//8)
    latent = torch.rand(1, 3, img_size[0]*8, img_size[1]*8)
    
    jit_model = torch.jit.trace(encoder, latent)
    output_path = args.output_name + "/" + "vae_encoder"
    os.makedirs(output_path, exist_ok=True)
    encoder_model = output_path + '/' + "vae_encoder" + '.pt'
    jit_model.save(encoder_model)
    log.info(f"end vae encoder saved to {encoder_model}")


def export_vaedecoder():
    vae = pipe.vae
    log.info(f" start vae decoder convert")
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder2 = vae.post_quant_conv
            self.decoder1 = vae.decoder
        
        def forward(self, x):
            x = self.decoder2(x)
            x = self.decoder1(x)
            return x

    decoder = Decoder()
    img_size = (512, 512)
    latent = torch.rand(1,4, img_size[0]//8, img_size[1]//8)
    jit_model = torch.jit.trace(decoder, latent)
    output_path = args.output_name + "/" + "vae_decoder"
    os.makedirs(output_path, exist_ok=True)
    decoder_model = output_path + '/' + "vae_decoder" + '.pt'
    jit_model.save(decoder_model)
    log.info(f"end vae decoder {decoder_model}")

def run_unet():
    if args.unet_safetensors_path is not None:
        log.info(f" start unet convert to pt")
        output_path = args.output_name + "/unet"
        os.makedirs(output_path, exist_ok=True)
        if args.lora_path is not None:
            log.info(f" start unet fuse lora convert")
            unet_output_name = output_path + "/unet_fuse_{}.pt".format(args.batch)
        else:
            log.info(f" start unet convert")
            unet_output_name = output_path + "/unet_{}.pt".format(args.batch)
        jit_model = torch.jit.trace(myunet, create_unet_input())
        jit_model.save(unet_output_name)
        log.info(f" end unet convert")

def run_text_encoder():
    if args.unet_safetensors_path is not None:
        log.info(f" start text encoder convert")
        jitmodel = torch.jit.trace(myte, create_te_input())
        output_path = args.output_name + "/text_encoder"
        os.makedirs(output_path, exist_ok=True)
        text_encoder_onnx_path = output_path + "/text_encoder.onnx"
        torch.onnx.export(jitmodel, create_te_input(), text_encoder_onnx_path, opset_version=11, verbose=False)
        log.info(f" end  text encoder convert")

def run_controlnet():
    if args.controlnet_path is not None:
        log.info(f" start controlnet convert")
        jit_model = torch.jit.trace(mycontrolnet, create_controlnet_input())
        output_path = args.output_name + "/controlnet"
        os.makedirs(output_path, exist_ok=True)
        controlnet_output_name = output_path + "/controlnet2_{}.pt".format(args.batch)
        torch.onnx.export(jit_model, create_controlnet_input(), controlnet_output_name, opset_version=11, verbose=False)
        # jit_model.save(controlnet_output_name)
        log.info(f" end controlnet convert")

def run_vae():
    if args.unet_safetensors_path is not None:
        log.info(f" start vae convert")
        export_vaencoder()
        export_vaedecoder()
        log.info(f" end vae convert")

log.info(f"start convert")
run_controlnet()
run_unet()
run_text_encoder()
run_vae()
log.info(f"all done")