'''example
    "awportrait": {
        "name": "awportrait",
        "encoder": "te_f32.bmodel",
        "unet": "unet_2_1684x_F16.bmodel",
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel",
        "controlnet": "canny_multize",
        "latent_shape": {
            "512x512": True,
            "768x512": True,
            "512x768": False
        }
    },
'''

import json
import os

models_path =  "models/basic"

folders_name = os.listdir(models_path)

def build_json():
    data = {}
    for i in folders_name:
        # data[i]
        data[i] = {
            "name": i,
            "encoder": "sdv15_text.bmodel",
            "unet": "sdv15_unet_multisize.bmodel",
            "vae_decoder": "sdv15_vd_multisize.bmodel",
            "vae_encoder": "sdv15_ve_multisize.bmodel",
            "controlnet": "canny_multize",
            "latent_shape": {
                "512x512": "True",
                "768x512": "True",
                "512x768": "False"
            }
        }

    # print(data)
    dict_str = json.dumps(data, ensure_ascii=False, indent=4)
    print("model_path = " + dict_str)


build_json()