import os
import os.path as path
import torch
import numpy as np
from safetensors.torch import load_file
from safetensors.torch import save_file
import argparse
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--safetensors_path', type=str)
parser.add_argument('-l', '--lcm_lora_path', type=str, default= "./lcm-lora-sdv1-5")
parser.add_argument('-u', '--unet_pt_path', type=str)
parser.add_argument('-t', '--text_encoder_onnx_path', type=str)
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_single_file(args.safetensors_path, load_safety_checker=False)

for para in pipe.unet.parameters():
    para.requires_grad = False

for para in pipe.text_encoder.parameters():
    para.requires_grad = False

pipe.load_lora_weights(args.lcm_lora_path)
pipe.fuse_lora()

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


def create_unet_input():
    img_size = (512, 512)
    batch = 1
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

if args.unet_pt_path is not None:
    jit_model = torch.jit.trace(myunet, create_unet_input())
    jit_model.save(args.unet_pt_path)


def myte(te_input):
    ret = pipe.text_encoder(te_input)
    return ret.last_hidden_state

def create_te_input():
    return torch.Tensor([[0]*77]).long()


if args.text_encoder_onnx_path is not None:
    jitmodel = torch.jit.trace(myte, create_te_input())
    torch.onnx.export(jitmodel, create_te_input(), args.text_encoder_onnx_path, opset_version=11, verbose=False)
