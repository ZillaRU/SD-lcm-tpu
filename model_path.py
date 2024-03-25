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
    "babes": {
        "name": "babes",
        "encoder": "te_f32.bmodel",
        "unet": {"512": "unet_1_1684x_f16_attention.bmodel",
                 "768": ""},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel"
    }
}
