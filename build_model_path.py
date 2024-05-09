'''example
    "awportrait": {
            "name": "",
            "encoder": "",
            "unet": {"512": "unet_2_1684x_F16.bmodel",
                    "768": ""
            },
            "vae_decoder": "vae_decoder_multize.bmodel",
            "vae_encoder": "vae_encoder_multize.bmodel"
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
            "unet": {"512": "sdv15_unet_multisize.bmodel",
                     "768": ""
                    },
            "vae_decoder": "sdv15_vd_multisize.bmodel",
            "vae_encoder": "sdv15_ve_multisize.bmodel"
        }

    # print(data)

    print(json.dumps(data, ensure_ascii=False, indent=4))


build_json()