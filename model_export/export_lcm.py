import os
import os.path as path
import torch
import numpy as np
from safetensors.torch import load_file
from safetensors.torch import save_file

from diffusers import StableDiffusionPipeline

from tqdm import tqdm

stablediffusion_checkpoint = "./st_models/stablediffusion/awportrait_v13.safetensors"
pipe = StableDiffusionPipeline.from_single_file(stablediffusion_checkpoint, load_safety_checker=False)

def get_layer_by_name(name):
    name = name.replace("lora_unet_", "")
    res = []
    is_digit = False
    name_list = name.split("_")
    total = ["ff","attn2","attn1", "attn3", "mid_block"]
    for aname in name_list:
        if aname.isdigit():
            res.append(aname)
            is_digit = True
        else:
            if len(res) == 0:
                res.append(aname)
            else:
                if is_digit:
                    res.append(aname)
                    is_digit = False
                else:
                    if res[-1] in total:
                        res.append(aname)
                    else:
                        res[-1] += "_" + aname
    
    cur = pipe.unet
    for i in res:
        if i.isdigit():
            cur = cur[int(i)]
        else:
            if not hasattr(cur, i):
                import pdb;pdb.set_trace()
            cur = getattr(cur, i)
    return cur

for para in pipe.unet.parameters():
    para.requires_grad = False

# safetensors weight 
# sdxl_lora = load_file("/workspace/demos/tpukern/test/mw/coeffex/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors")
# HEAD="lora_unet_"
# total_lora_keys = set([i.split(".lora_")[0] for i in sdxl_lora.keys() if "alpha" not in i])

# for key in tqdm(total_lora_keys):
#     layer = get_layer_by_name(key)
#     data  = layer.weight
#     np.save("./unet15/before/" + key + ".npy", data.cpu().numpy())

#     # lora_unet_down_blocks_0_attentions_0_proj_in.alpha
#     # lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
#     # lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight
#     temp = {}
#     temp['alpha'] = sdxl_lora[key + ".alpha"]
#     temp['lora_down'] = sdxl_lora[key + ".lora_down.weight"]
#     temp['lora_up'] = sdxl_lora[key + ".lora_up.weight"]
#     np.savez("./unet15/lora/" + key + ".npz", **temp)


# have_matched = []
# need_save = []
# fix_key = None
# matched = {}

adapter_id = "latent-consistency/lcm-lora-sdv1-5"
pipe.load_lora_weights("/data/aigc/demos/tpukern/test/mw/coeffex/lcm-lora-sdv1-5")

pipe.fuse_lora()
# for key in tqdm(total_lora_keys):
#     layer = get_layer_by_name(key)
#     data  = layer.weight
#     np.save("./unet15/after/" + key + ".npy", data.cpu().numpy())

# import pdb;pdb.set_trace()



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

fake_input = (latent_model_input, t, prompt_embeds, mid_block_additional_residual, *down_block_additional_residuals)

# config  



jit_model = torch.jit.trace(myunet, fake_input)
jit_model.save("awportrait_v13_unet_lcm_patch1.pt")
