import random
import os
import numpy as np
import torch
from PIL import Image
import json
def create_size(*size_elements):
    unique_size_elements = sorted(list(set(size_elements)))
    all_sizes = []
    for i in unique_size_elements:
        for j in unique_size_elements:
            all_sizes.append([i, j])
    return [ (f"{size[0]}:{size[1]}", size) for size in all_sizes]


def seed_torch(seed=1029):
    seed = seed % 4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)


def ratio_resize(source_img, target_size):
    old_size = list(source_img.size)  # (width, height)
    if target_size != old_size:
        ratio = min(float(target_size[i]) / old_size[i] for i in range(len(old_size)))
        new_size = tuple(int(i * ratio) for i in old_size)

        img = source_img.resize(new_size)
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))

        return new_img
    else:
        return source_img

def get_model_path():
    model_path = {}
    models_path = "models/basic"
    folders_name = os.listdir(models_path)
    for i in folders_name:
        model_path[i] = {
            "name": i,
            "encoder": "sdv15_text.bmodel",
            "unet": "sdv15_unet_multisize.bmodel",
            "vae_decoder": "sdv15_vd_multisize.bmodel",
            "vae_encoder": "sdv15_ve_multisize.bmodel",
        }
    # dict_str = json.dumps(model_path, ensure_ascii=False, indent=4)
    # print("model_path = " + dict_str)
    return model_path

def get_model_input_info(model_stage_info_dict):
    model_input_shape_list = []
    for i in model_stage_info_dict:
        tmp_list = []
        input_shape = i["input_tensor"][0]["data_shape"]
        tmp_list.append(input_shape[3] * 8)  # W
        tmp_list.append(input_shape[2] * 8)  # H
        model_input_shape_list.append(tmp_list)
    return model_input_shape_list





