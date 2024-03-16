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
                 "768": "unet2.bmodel"},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel"
    },
    "RealCartoon2.5D": {
        "name": "RealCartoon2.5D",
        "encoder": "te.bmodel",
        "unet": {"512": "unet.bmodel",
                 "768": "unet2.bmodel"},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel"

    },
    "majicMIX_realistic": {
        "name": "majicMIX_realistic",
        "encoder": "te.bmodel",
        "unet": {"512": "unet.bmodel",
                 "768": "unet2.bmodel"},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel"
    },
    "majicMIX_lux": {
        "name": "majicMIX_lux",
        "encoder": "te.bmodel",
        "unet": {"512": "unet.bmodel",
                 "768": "unet2.bmodel"},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel"
    },
    "majicMIX_fantasy": {
        "name": "majicMIX_fantasy",
        "encoder": "te.bmodel",
        "unet": {"512": "unet.bmodel",
                 "768": "unet2.bmodel"},
        "vae_decoder": "vae_decoder_multize.bmodel",
        "vae_encoder": "vae_encoder_multize.bmodel"
    }
}
