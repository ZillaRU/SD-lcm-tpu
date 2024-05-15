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

model_path = {
    "hello": {
        "name": "hellonijicute",
        "encoder": "sdv15_text.bmodel",
        "unet": "sdv15_unet_multisize.bmodel",
        "vae_decoder": "sdv15_vd_multisize.bmodel",
        "vae_encoder": "sdv15_ve_multisize.bmodel",
        "latent_shape": {
            "512x512": "True",
            "768x512": "False",
            "512x768": "False"
        }

    },
    "awportrait": {
        "name": "awportrait",
        "encoder": "te_f32.bmodel",
        "unet": "unet_2_1684x_F16.bmodel",
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel",
        "latent_shape": {
            "512x512": "True",
            "768x512": "False",
            "512x768": "False"
        }
    },
    "meinamix":{
        "name": "meinamix",
        "encoder": "sdv15_te.bmodel",
        "unet": "unet_1_1684x_F16.bmodel",
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel",
        "controlnet": "canny_multize",
        "latent_shape": {
            "512x512": "True",
            "768x512": "False",
            "512x768": "False"
        }
    },
    "toonyou":{
        "name": "toonyou",
        "encoder": "sdv15_text.bmodel",
        "unet": "sdv15_unet_multisize.bmodel",
        "vae_decoder": "sdv15_vd_multisize.bmodel",
        "vae_encoder": "sdv15_ve_multisize.bmodel",
        "latent_shape": {
            "512x512": "True",
            "768x512": "True",
            "512x768": "True"
        }
    },
}
