import os
import os.path as path
import torch
import numpy as np
import argparse
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--safetensors_path', type=str)
parser.add_argument('-p', '--pt_dir', type=str)
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_single_file(args.safetensors_path, load_safety_checker=False)

def export_vaencoder():
    for para in pipe.vae.parameters():
        para.requires_grad = False

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
    encoder_model = args.pt_dir + '/' + "vae_encoder" + '.pt'
    os.makedirs(args.pt_dir, exist_ok=True)
    jit_model.save(encoder_model)
    log.info(f"vae encoder saved to {encoder_model}")


def export_vaedecoder():
    for para in pipe.vae.parameters():
        para.requires_grad = False
    vae = pipe.vae

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
    decoder_model = args.pt_dir + '/' + "vae_decoder" + '.pt'
    os.makedirs(args.pt_dir, exist_ok=True)
    jit_model.save(decoder_model)
    log.info(f"vae decoder saved to {decoder_model}")
