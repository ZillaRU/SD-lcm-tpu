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
    "awportrait": {
        "name": "awportrait",
        "encoder": "te_f32.bmodel",
        "unet": "unet_2_1684x_F16.bmodel",
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel",
        "controlnet": "canny_multize",
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
    }
}
