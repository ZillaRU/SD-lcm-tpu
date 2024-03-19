import os
import os.path as path
import torch
import numpy as np
import argparse
import yaml
from collections import defaultdict
import logging 
import sys
from safetensors.torch import load_file
from safetensors.torch import save_file

from diffusers import StableDiffusionPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt

from transformers import (
    AutoFeatureExtractor,
    BertTokenizerFast,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)
from diffusers.models import UNet2DConditionModel
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}
    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]
    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint

def load_safetensors(path):
    from safetensors import safe_open
    checkpoint = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
    return checkpoint



name = {
    "1.5": "openai/clip-vit-large-patch14",
    "1.4": "openai/clip-vit-large-patch14",
}


def add_text_encoder(kv, sdversion="1.5"):
    key_prefix = 'cond_stage_model'
    if sdversion not in name:
        raise ValueError("sdversion error")
    repo_name = name[sdversion]
    text_encoder = CLIPTextModel.from_pretrained(repo_name)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    for p_name, para in text_encoder.named_parameters():
        kv["cond_stage_model.transformer."+p_name] = para
        # print(p_name)
    kv["cond_stage_model.transformer.text_model.embeddings.position_ids"] = torch.tensor([list(range(77))], dtype=torch.float16)
    return kv

def check_has_text_encoder(kv):
    key_prefix = 'cond_stage_model'
    flag = False
    for k in kv.keys():
        if key_prefix in k:
            flag = True
    if flag:
        return False, kv
    kv = add_text_encoder(kv)
    return True, kv

def load_ckpt(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint

def read_file(path):
    if path.endswith(".safetensors"):
        return load_safetensors(path)
    elif path.endswith(".ckpt"):
        return load_ckpt(path)
    else:
        raise ValueError(f"Unknown file type: {path}")

def check_is_ok(path, self_save=True):
    kv = read_file(path)
    responsiblity = [check_has_text_encoder]
    flag = False
    for fn in responsiblity:
        cflag, kv = fn(kv)
        if cflag:
            flag = True
    if flag:
        if self_save:
            save_kv(kv, path)
            print("save ok")

def save_kv(kv, path):
    # safetensors 
    if path.endswith(".safetensors"):
        save_file(kv,path)
    elif path.endswith(".ckpt"):
        torch.save({"state_dict":kv}, path)
    else:
        raise ValueError(f"Unknown file type: {path}")


text_embedding_dim = {
    "1.4": 768,
    "1.5": 768,
    "2.1": 1024,
}

class ModelExporter:
    def __init__(self, args, stablediffusion_checkpoint=None, lora_checkpoint=None, controlnet_checkpoint=None, 
            batch=2, if_trace=True, vae_checkpoint=None):
        self.args=args
        self.stablediffusion_checkpoint = stablediffusion_checkpoint
        self.lora_checkpoint = lora_checkpoint
        self.controlnet_checkpoint = controlnet_checkpoint
        self.batch = batch
        self.if_trace = if_trace
        self.original_config_file = 'st_models/controlnet/config.yaml'  # https://github.com/huggingface/diffusers/pull/2593
        self.pipe = None
        self.version = args.sdversion
        self.controlnet = None
        self.trace_unet = False
        self.trace_controlnet = False
        self.unet_dir = 'pt_models/{}/'.format(self.args.cn)
        self.controlnet_dir = 'pt_models/{}/'.format(self.args.cn)
        self.unet_name = ''
        self.text_embedding_dim = text_embedding_dim[self.version]
        self.controlnet_name = ''
        self.unet_model = ''
        self.encoder_model = ''
        self.decoder_model = ''
        self.controlnet_model = ''
        self.img_size = None
        self.trace_text_encoder = False
        self.trace_vae = False
        self.vae_checkpoint = vae_checkpoint
        self.vae_config = "vae_config.yaml"
    
    def model_init(self):
        assert self.version in ["1.4",'1.5','2.1']
        if self.version == "2.1":
            self.original_config_file = 'st_models/controlnet/config21.yaml' # https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v21.yaml
        if self.stablediffusion_checkpoint is not None:
            log.info('load stablediffusion model')
            self.trace_unet = True
            self.unet_name = self.args.base_name + '-' + self.args.sdname
            check_is_ok(self.stablediffusion_checkpoint)
            print(self.stablediffusion_checkpoint)
            self.pipe = StableDiffusionPipeline.from_ckpt(self.stablediffusion_checkpoint,load_safety_checker=False)
            # self.pipe = load_checkpoint(checkpoint_path, config)
            if self.lora_checkpoint is not None:
                self.unet_name += '-lora_' + self.args.lrname
                self.pipe = self.load_lora_weights('cpu', torch.float32)
            for para in self.pipe.unet.parameters():
                para.requires_grad = False
            self.trace_text_encoder = True
            self.trace_vae = True
        self.load_vae_weight()
        if self.args.only_vae:
            self.export_vaencoder()
            self.export_vaedecoder()
            print("export vae encoder and decoder done and exit")
            sys.exit(0)

        if self.controlnet_checkpoint is not None:
            log.info('load controlnet model')
            self.trace_controlnet = True
            self.controlnet_name = self.args.base_name + '-' + self.args.cnname + f'-{self.batch}'
            self.controlnet = download_controlnet_from_original_ckpt(
                checkpoint_path=self.controlnet_checkpoint,
                original_config_file=self.original_config_file,
                from_safetensors=True,
                device='cpu')
            for para in self.controlnet.parameters():
                para.requires_grad = False
        
    def load_lora_weights(self, device, dtype, weight=1):
        import pdb; pdb.set_trace()
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(self.lora_checkpoint, device=device)
        mystr = ""
        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
            layer, elem = key.split('.', weight)
            updates[layer][elem] = value
            mystr += key + '\n'
            print(key)
        f = open("test.txt", "w")
        f.write(mystr)
        f.close()
        # directly update weight in diffusers model
        for layer, elems in updates.items():
            is_unet = False
            if "text" in layer:
                is_unet = False
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.pipe.text_encoder
                
                # log.warning(f"update text encoder layer {layer_infos}")
            else:
                is_unet = True
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.pipe.unet
                # log.info(f"update unet layer {layer_infos}")

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0
            
            if is_unet:
                print(curr_layer)
                np.save("./unet_layer/"+layer+".npy", curr_layer.weight.data.numpy())
            
            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        return self.pipe

    def load_vae_weight(self):
        if self.vae_checkpoint is not None and self.vae_checkpoint !="":
            pass
        else:
            return 
        print("start load vae weight ....... ")
        config = self.vae_config
        checkpoint_path = self.vae_checkpoint
        original_config = OmegaConf.load(config)
        image_size = 512
        checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]
        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)
        self.pipe.vae.load_state_dict(converted_vae_checkpoint)
        print("load vae weight done ....... ")

    def mycontrolnet(self, controlnet_latent_model_input, controlnet_prompt_embeds, image, t):
        ret1, ret2= self.controlnet(
            controlnet_latent_model_input,
            t,
            controlnet_prompt_embeds,
            controlnet_cond = image,
            conditioning_scale=1,
            guess_mode=False,
            return_dict=False,
        )
        return ret1[0], ret1[1], ret1[2], ret1[3], ret1[4], ret1[5],  ret1[6], ret1[7], ret1[8], ret1[9], ret1[10], ret1[11],  ret2
    
    def myencoder(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        ):
        class_labels = None
        timestep_cond = None
        attention_mask = None
        cross_attention_kwargs = None

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.pipe.unet.config.center_input_sample:
            sample = 2 * sample - 1.0
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.pipe.unet.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.pipe.unet.dtype)

        emb = self.pipe.unet.time_embedding(t_emb, timestep_cond)

        if self.pipe.unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.pipe.unet.config.class_embed_type == "timestep":
                class_labels = self.pipe.unet.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.pipe.unet.class_embedding(class_labels).to(dtype=self.pipe.unet.dtype)

            if self.pipe.unet.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.pipe.unet.config.addition_embed_type == "text":
            aug_emb = self.pipe.unet.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.pipe.unet.time_embed_act is not None:
            emb = self.pipe.unet.time_embed_act(emb)

        if self.pipe.unet.encoder_hid_proj is not None:
            encoder_hidden_states = self.pipe.unet.encoder_hid_proj(encoder_hidden_states)
        
        # 2. pre-process
        sample = self.pipe.unet.conv_in(sample)

        # 3. down
        down_block_res_samples = [sample,]
        for downsample_block in self.pipe.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        
        # 4. mid
        if self.pipe.unet.mid_block is not None:
            sample = self.pipe.unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        
        return down_block_res_samples[0], down_block_res_samples[1], down_block_res_samples[2], down_block_res_samples[3], down_block_res_samples[4], down_block_res_samples[5], \
            down_block_res_samples[6], down_block_res_samples[7], down_block_res_samples[8], down_block_res_samples[9], down_block_res_samples[10], down_block_res_samples[11], \
            sample, emb
    
    def mydecoder(
            self,
            sample,
            encoder_hidden_states,
            emb,
            mid_block_additional_residual, # 1
            down_block_res_samples_0,# 12
            down_block_res_samples_1,# 12
            down_block_res_samples_2,# 12
            down_block_res_samples_3,# 12
            down_block_res_samples_4,# 12
            down_block_res_samples_5,# 12
            down_block_res_samples_6,# 12
            down_block_res_samples_7,# 12
            down_block_res_samples_8,# 12
            down_block_res_samples_9,# 12
            down_block_res_samples_10,# 12
            down_block_res_samples_11,# 12
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
        # 1. trace 得到的模型是否正确 
        # 2. mlir 是否支持这种编译 
        # 3. 对性能有没有什么影响 
        down_block_res_samples = [down_block_res_samples_0, down_block_res_samples_1, down_block_res_samples_2, down_block_res_samples_3, down_block_res_samples_4, down_block_res_samples_5, down_block_res_samples_6, down_block_res_samples_7, down_block_res_samples_8, down_block_res_samples_9, down_block_res_samples_10, down_block_res_samples_11]
        down_block_additional_residuals = [down_block_additional_residuals_0, down_block_additional_residuals_1, down_block_additional_residuals_2, down_block_additional_residuals_3, down_block_additional_residuals_4, down_block_additional_residuals_5, down_block_additional_residuals_6, down_block_additional_residuals_7, down_block_additional_residuals_8, down_block_additional_residuals_9, down_block_additional_residuals_10, down_block_additional_residuals_11]
        attention_mask = None
        cross_attention_kwargs = None

        default_overall_up_factor = 2**self.pipe.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples   
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        # 5. up
        for i, upsample_block in enumerate(self.pipe.unet.up_blocks):
            is_final_block = i == len(self.pipe.unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.pipe.unet.conv_norm_out:
            sample = self.pipe.unet.conv_norm_out(sample)
            sample = self.pipe.unet.conv_act(sample)
        sample = self.pipe.unet.conv_out(sample)

        return sample

    def myunet(
        self,
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
        ret = self.pipe.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )

        return ret.sample
    
    def my_only_unet(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        ):
        ret = self.pipe.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        return ret.sample    

    
    def export_textencoder(self):
        
        def mytextencoder(self, inputs):
            res = self.pipe.text_encoder(
                inputs,
                return_dict=False
            )
            return res
           
        if not self.trace_text_encoder:
            return 
        for para in self.pipe.text_encoder.parameters():
            para.requires_grad = False
        batch = 1
        fake_input = torch.randint(0, 1000, (batch, 77))
        self.encoder_model = self.unet_dir + '/' + "encoder.onnx"
        os.makedirs(path.dirname(self.encoder_model), exist_ok=True)
        torch.jit.trace(mytextencoder, fake_input).save(self.unet_dir + '/' + "encoder.pt")
        torch.onnx.export(self.pipe.text_encoder, fake_input, self.encoder_model, verbose=False, opset_version=14, input_names=["input_ids"], output_names=["output"])
        log.info(f'textencoder model saved to {self.encoder_model}')
        temp1 = """model_transform.py \
        --model_name encoder \
        --input_shape {} \
        --model_def encoder.onnx \
        --mlir encoder.mlir\n"""
        temp2 = """model_deploy.py \
        --mlir encoder.mlir \
        --quantize F32 \
        --chip bm1684x \
        --model text_encoder_1684x_f32.bmodel\n"""
        convert1 = temp1.format(str([[batch, 77]]).replace(" ",""))
        convert2 = temp2
        self.encoder_convert = self.unet_dir + '/' + "text_encoder_convert.sh"
        with open(self.encoder_convert, 'w') as f:
            f.write(convert1)
            f.write(convert2)
        log.info(f'encoder convert script saved to {self.encoder_convert}')

    def export_controlnet(self):
        if not self.trace_controlnet:
            log.warning('controlnet is not traced')
            return
        img_size = self.img_size or (512, 512)
        controlnet_latent_model_input = torch.rand(self.batch, 4, img_size[0]//8, img_size[1]//8)
        controlnet_prompt_embeds = torch.rand(self.batch, 77, self.text_embedding_dim)
        image = torch.rand(self.batch, 3, *img_size)
        t = torch.tensor([999])
        fake_input = (controlnet_latent_model_input, controlnet_prompt_embeds, image, t)
        self.controlnet_model = self.controlnet_dir + self.controlnet_name + '.pt'

        dir_name = path.dirname(self.controlnet_model)
        if not path.exists(dir_name):
            os.makedirs(dir_name)
        if self.if_trace:
            jit_model = torch.jit.trace(self.mycontrolnet, fake_input)
            jit_model.save(self.controlnet_model)
        log.info(f'controlnet model saved to {self.controlnet_model}')
        self.controlnet_convert = self.controlnet_dir + '/' + "controlnet_convert.sh"
        template1 = """model_transform.py \
        --model_name controlnet_{} \
        --input_shape {} \
        --model_def {}.pt \
        --mlir controlnet_2_{}.mlir\n"""
        template2 = """model_deploy.py \
        --mlir controlnet_2_{}.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model {}.bmodel\n"""
        name2 = "controlnet"
        fake_input_shapes = [list(i.shape) for i in fake_input]
        convert1 = template1.format(self.batch, str(fake_input_shapes).replace(" ",""), self.controlnet_name, self.batch)
        convert2 = template2.format(self.batch, name2)
        
        with open(self.controlnet_convert, 'w') as f:
            f.write(convert1)
            f.write(convert2)

    def export_only_unet(self):
        # do not provide controlnet data 
        pass

    def export_unet(self):
        if not self.trace_unet:
            log.error('unet is not traced')
            return
        log.info('unet is traced')
        img_size = self.img_size or (512, 512)
        latent_model_input = torch.rand(self.batch, 4, img_size[0]//8, img_size[1]//8)
        t = torch.tensor([999])
        prompt_embeds = torch.rand(self.batch, 77, self.text_embedding_dim)
        mid_block_additional_residual = torch.rand(self.batch, 1280, img_size[0]//64, img_size[1]//64)
        down_block_additional_residuals = []
        down_block_additional_residuals.append(torch.rand(self.batch, 320, img_size[0]//8, img_size[1]//8))
        down_block_additional_residuals.append(torch.rand(self.batch, 320, img_size[0]//8, img_size[1]//8))
        down_block_additional_residuals.append(torch.rand(self.batch, 320, img_size[0]//8, img_size[1]//8))
        down_block_additional_residuals.append(torch.rand(self.batch, 320, img_size[0]//16, img_size[1]//16))
        down_block_additional_residuals.append(torch.rand(self.batch, 640, img_size[0]//16, img_size[1]//16))
        down_block_additional_residuals.append(torch.rand(self.batch, 640, img_size[0]//16, img_size[1]//16))
        down_block_additional_residuals.append(torch.rand(self.batch, 640, img_size[0]//32, img_size[1]//32))
        down_block_additional_residuals.append(torch.rand(self.batch, 1280, img_size[0]//32, img_size[1]//32))
        down_block_additional_residuals.append(torch.rand(self.batch, 1280, img_size[0]//32, img_size[1]//32))
        down_block_additional_residuals.append(torch.rand(self.batch, 1280, img_size[0]//64, img_size[1]//64))
        down_block_additional_residuals.append(torch.rand(self.batch, 1280, img_size[0]//64, img_size[1]//64))
        down_block_additional_residuals.append(torch.rand(self.batch, 1280, img_size[0]//64, img_size[1]//64))
        fake_input = (latent_model_input, t, prompt_embeds, mid_block_additional_residual, *down_block_additional_residuals)
        unet_name = self.unet_name + f'-unet-{self.batch}'
        self.unet_model = self.unet_dir + unet_name + '.pt'
        dir_name = path.dirname(self.unet_model)
        if not path.exists(dir_name):
            os.makedirs(dir_name)
        
        jit_model = torch.jit.trace(self.myunet, fake_input)
        jit_model.save(self.unet_model)

        log.info(f'unet model saved to {self.unet_model}')
        self.unet_convert = self.unet_dir +'/' + "unet_convert.sh"
        template1 = """model_transform.py \
        --model_name unet_{} \
        --input_shape {} \
        --model_def {}.pt \
        --mlir unet_2_{}.mlir\n"""
        template2 = """model_deploy.py \
        --mlir unet_{}_{}.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --model unet_{}_1684x_f16.bmodel\n"""
        fake_input_shapes = [list(i.shape) for i in fake_input]
        convert1 = template1.format(self.batch, str(fake_input_shapes).replace(" ",""), unet_name, img_size[0])
        convert2 = template2.format(self.batch, img_size[0], self.batch, img_size)
        with open(self.unet_convert, 'w') as f:
            f.write(convert1)
            f.write(convert2)
        log.info(f'unet convert script saved to {self.unet_convert}')

    def export_vaencoder(self):
        if not self.trace_vae:
            log.error('vae is not traced')
            return
        for para in self.pipe.vae.parameters():
            para.requires_grad = False

        vae = self.pipe.vae

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
        img_size = self.img_size or (512, 512)
        img_size = (img_size[0]//8, img_size[1]//8)
        latent = torch.rand(1,3, img_size[0]*8, img_size[1]*8)
        jit_model = torch.jit.trace(encoder, latent)
        self.encoder_model = self.unet_dir + '/' + "vae_encoder" + '.pt'
        os.makedirs(path.dirname(self.encoder_model), exist_ok=True)
        jit_model.save(self.encoder_model)
        log.info("vae encoder saved to vae_encoder.pt")

        temp1 = """model_transform.py \
        --model_name vae_encoder \
        --input_shape {} \
        --model_def vae_encoder.pt \
        --mlir vae_encoder.mlir\n"""
        temp2 = """model_deploy.py \
        --mlir vae_encoder.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --model vae_encoder_1684x_f16.bmodel\n"""
        convert1 = temp1.format(str([[1,3,img_size[0]*8, img_size[1]*8]]).replace(" ",""))
        convert2 = temp2
        self.encoder_convert = self.unet_dir + '/' + "vae_encoder_convert.sh"
        with open(self.encoder_convert, 'w') as f:
            f.write(convert1)
            f.write(convert2)

    def export_vaedecoder(self):
        if not self.trace_vae:
            log.error('vae is not traced')
            return
        for para in self.pipe.vae.parameters():
            para.requires_grad = False
        vae = self.pipe.vae
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
        img_size = self.img_size or (512, 512)
        latent = torch.rand(1,4, img_size[0]//8, img_size[1]//8)
        jit_model = torch.jit.trace(decoder, latent)
        self.decoder_model = self.unet_dir + '/' + "vae_decoder" + '.pt'
        os.makedirs(path.dirname(self.decoder_model), exist_ok=True)
        jit_model.save(self.decoder_model)
        log.info("vae decoder saved to vae_decoder.pt")
        temp1 = """model_transform.py \
        --model_name vae_decoder \
        --input_shape {} \
        --model_def vae_decoder.pt \
        --mlir vae_decoder.mlir\n"""
        temp2 = """model_deploy.py \
        --mlir vae_decoder.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --model vae_decoder_1684x_f16.bmodel\n"""
        convert1 = temp1.format(str([[1,4, img_size[0]//8, img_size[1]//8]]).replace(" ",""))
        convert2 = temp2
        self.decoder_convert = self.unet_dir + '/' + "vae_decoder_convert.sh"
        with open(self.decoder_convert, 'w') as f:
            f.write(convert1)
            f.write(convert2)
        log.info("vae decoder convert script saved to vae_decoder_convert.sh")

    def export(self):
        log.info("Start Exporting")
        log.info("init model")
        self.model_init()
        self.export_controlnet()
        self.export_unet()
        self.export_textencoder()
        self.export_vaencoder()
        self.export_vaedecoder()
        log.info("Finish Exporting")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Model Transformation and Compile Settings."
    )
    parser.add_argument(
        "--base_name", type=str, 
        default='1.5', 
        help="Stable Diffusion version: 1.4 or 1.5 or 2.1"
    )
    parser.add_argument(
        "--sdname", type=str, 
        default="mname", 
        help="The name of SD Model"
    )
    parser.add_argument(
        "--lrname", type=str, 
        default="lrname", 
        help="The name of LoRA"
    )
    parser.add_argument(
        "--cnname", type=str, 
        default="cname", 
        help="The name of Controlnet with SD version, like `controlnet11Models_normal_1.5`"
    )
    parser.add_argument(
        "--cn", type=str, default="123", help="current name"
    )
    parser.add_argument(
        "--batch", type=int, default=2, help="The batch number of input."
    )

    parser.add_argument(
        "--vae", type=str, default="none", help="The batch number of input."
    )
    
    parser.add_argument(
        "--pure_unet", type=bool, default=False, help="only unet no controlnet"
    )

    parser.add_argument(
        "--only_vae", type=bool, default=False, help="The batch number of input."
    )

    parser.add_argument(
        "--if_trace", type=bool, default=True, help="The batch number of input."
    )

    parser.add_argument(
        "--sdversion", type=str, default="1.5", help="stable diffusion version"
    )

    args = parser.parse_args()
    stablediffusion_path = args.sdname
    if "." in stablediffusion_path:
        stablediffusion_path = "./st_models/stablediffusion/" + stablediffusion_path
    elif stablediffusion_path != "none":
        stablediffusion_path = "./st_models/stablediffusion/" + stablediffusion_path 
    else:
        stablediffusion_path = None
    lora_path = args.lrname
    if lora_path =="none":
        lora_path = None
    else:
        lora_path = "./st_models/lora/" + lora_path
    
    controlnet_path = args.cnname
    if controlnet_path == "none":
        controlnet_path = None
    else:
        controlnet_path = "./st_models/controlnet/" + controlnet_path

    exporter = ModelExporter(args, stablediffusion_path, lora_path, controlnet_path, 2, args.if_trace)
    exporter.export()