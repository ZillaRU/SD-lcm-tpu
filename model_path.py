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

model_path = {
    "awportrait": {
        "name": "awportrait",
        "encoder": "te_f32.bmodel",
        "unet": {"512": "unet_2_1684x_F16.bmodel",
                 "768": ""},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel",
        "controlnet": "canny_multize"
    },
    "meinamix":{
        "name": "meinamix",
        "encoder": "sdv15_te.bmodel",
        "unet": {"512": "unet_1_1684x_F16.bmodel",
                 "768": ""},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel",
        "controlnet": "canny_multize"
    }
}
