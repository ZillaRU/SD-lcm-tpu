# current only support sd15
import warnings
import logging
import os
import torch
from diffusers import StableDiffusionPipeline
import time
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)



log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ConvertorPtOnnx():
    def __init__(self, unet_path, controlnet_path=None, lora_path=None, unet_url=None, controlnet_url=None, lora_url=None, batch=None, version=None, merge=None, output_name=None, debug=None):
        self.unet_path = unet_path
        self.controlnet_path = controlnet_path
        self.lora_path = lora_path
        self.unet_url = unet_url,
        self.controlnet_url = controlnet_url
        self.lora_url = lora_url
        self.batch = batch
        self.version = version
        self.merge = merge
        self.debug = debug
        self.output_name = output_name
        self.unet_size = 512
        self.pipe = None


    def init_convertor(self):
        if self.debug:
            logging.getLogger('tensorflow').setLevel(logging.DEBUG)
            logging.getLogger("diffusers").setLevel(logging.DEBUG)
        assert self.version in ["sd15"], "only support sd15"
        assert self.batch in [1, 2], "only support batch 1 or 2"
        self.output_name = self.output_name + "_" + self.version

        os.makedirs(self.output_name, exist_ok=True)


    def init_pipe(self):
        if self.unet_path is not None:
            self.pipe = StableDiffusionPipeline.from_single_file(self.unet_path, load_safety_checker=False)
            if self.lora_path is not None:
                self.pipe.load_lora_weights(self.lora_path)
                self.pipe.fuse_lora()
            for para in self.pipe.unet.parameters():
                para.requires_grad = False
            for para in self.pipe.text_encoder.parameters():
                para.requires_grad = False
            for para in self.pipe.vae.parameters():
                para.requires_grad = False

        if self.controlnet_path is not None:
            original_config_file = "./safetensors/controlnet/config.yaml"
            self.pipe.controlnet = download_controlnet_from_original_ckpt(
                checkpoint_path=self.controlnet_path,
                original_config_file=original_config_file,
                from_safetensors=True,
                device="cpu"
            )
            for para in self.pipe.controlnet.parameters():
                para.requires_grad = False

    def myunet(self,
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
        down_block_additional_residuals = [down_block_additional_residuals_0, down_block_additional_residuals_1,
                                           down_block_additional_residuals_2, down_block_additional_residuals_3,
                                           down_block_additional_residuals_4, down_block_additional_residuals_5,
                                           down_block_additional_residuals_6, down_block_additional_residuals_7,
                                           down_block_additional_residuals_8, down_block_additional_residuals_9,
                                           down_block_additional_residuals_10, down_block_additional_residuals_11]
        ret = self.pipe.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )
        return ret.sample

    def create_controlnet_input(self):
        batch = self.batch
        img_size = (512, 512)
        controlnet_latent_model_input = torch.rand(batch, 4, img_size[0] // 8, img_size[1] // 8)
        controlnet_prompt_embeds = torch.rand(batch, 77, 768)
        image = torch.rand(batch, 3, img_size[0], img_size[1])
        t = torch.tensor([999])
        w = torch.tensor([1])
        return controlnet_latent_model_input, controlnet_prompt_embeds, image, t, w

    def mycontrolnet(self, controlnet_latent_model_input, controlnet_prompt_embeds, image, t, weight):
        ret1, ret2 = self.pipe.controlnet(
            controlnet_latent_model_input,
            t,
            controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=weight,
            guess_mode=False,
            return_dict=False,
        )
        return ret1[0], ret1[1], ret1[2], ret1[3], ret1[4], ret1[5], ret1[6], ret1[7], ret1[8], ret1[9], ret1[10], ret1[
            11], ret2

    def create_unet_input(self):
        img_size = (self.unet_size, self.unet_size)
        batch = self.batch
        latent_model_input = torch.rand(batch, 4, img_size[0] // 8, img_size[1] // 8)
        t = torch.tensor([999])
        prompt_embeds = torch.rand(batch, 77, 768)
        mid_block_additional_residual = torch.rand(batch, 1280, img_size[0] // 64, img_size[1] // 64)
        down_block_additional_residuals = []
        down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0] // 8, img_size[1] // 8))
        down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0] // 8, img_size[1] // 8))
        down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0] // 8, img_size[1] // 8))
        down_block_additional_residuals.append(torch.rand(batch, 320, img_size[0] // 16, img_size[1] // 16))
        down_block_additional_residuals.append(torch.rand(batch, 640, img_size[0] // 16, img_size[1] // 16))
        down_block_additional_residuals.append(torch.rand(batch, 640, img_size[0] // 16, img_size[1] // 16))
        down_block_additional_residuals.append(torch.rand(batch, 640, img_size[0] // 32, img_size[1] // 32))
        down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0] // 32, img_size[1] // 32))
        down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0] // 32, img_size[1] // 32))
        down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0] // 64, img_size[1] // 64))
        down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0] // 64, img_size[1] // 64))
        down_block_additional_residuals.append(torch.rand(batch, 1280, img_size[0] // 64, img_size[1] // 64))

        return (latent_model_input, t, prompt_embeds, mid_block_additional_residual, *down_block_additional_residuals)

    def myte(self, te_input):
        ret = self.pipe.text_encoder(te_input)
        return ret.last_hidden_state

    def create_te_input(self):
        return torch.Tensor([[0] * 77]).long()

    def export_vaencoder(self):
        log.info(f" start vae encoder convert")
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
        img_size = (512, 512)
        img_size = (img_size[0] // 8, img_size[1] // 8)
        latent = torch.rand(1, 3, img_size[0] * 8, img_size[1] * 8)

        jit_model = torch.jit.trace(encoder, latent)
        output_path = self.output_name + "/" + "vae_encoder"
        os.makedirs(output_path, exist_ok=True)
        encoder_model = output_path + '/' + "vae_encoder" + '.pt'
        jit_model.save(encoder_model)
        log.info(f"end vae encoder saved to {encoder_model}")

    def export_vaedecoder(self):
        vae = self.pipe.vae
        log.info(f" start vae decoder convert")

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
        img_size = (512, 512)
        latent = torch.rand(1, 4, img_size[0] // 8, img_size[1] // 8)
        jit_model = torch.jit.trace(decoder, latent)
        output_path = self.output_name + "/" + "vae_decoder"
        os.makedirs(output_path, exist_ok=True)
        decoder_model = output_path + '/' + "vae_decoder" + '.pt'
        jit_model.save(decoder_model)
        log.info(f"end vae decoder {decoder_model}")

    def run_unet(self):
        if self.unet_path is not None:
            log.info(f" start unet convert to pt")
            output_path = self.output_name + "/unet"
            os.makedirs(output_path, exist_ok=True)
            if self.lora_path is not None:
                log.info(f" start unet fuse lora convert")
                unet_output_name = output_path + "/unet_fuse_{}.pt".format(self.batch)
            else:
                log.info(f" start unet convert")
                unet_output_name = output_path + "/unet_{}.pt".format(self.batch)
            jit_model = torch.jit.trace(self.myunet, self.create_unet_input())
            jit_model.save(unet_output_name)
            log.info(f" end unet convert")

    def run_text_encoder(self):
        if self.unet_path is not None:
            log.info(f" start text encoder convert")
            jitmodel = torch.jit.trace(self.myte, self.create_te_input())
            output_path = self.output_name + "/text_encoder"
            os.makedirs(output_path, exist_ok=True)
            text_encoder_onnx_path = output_path + "/text_encoder.onnx"
            torch.onnx.export(jitmodel, self.create_te_input(), text_encoder_onnx_path, opset_version=11, verbose=False)
            log.info(f" end  text encoder convert")

    def run_controlnet(self):
        if self.controlnet_path is not None:
            log.info(f" start controlnet convert")
            jit_model = torch.jit.trace(self.mycontrolnet, self.create_controlnet_input())
            output_path = self.output_name + "/controlnet"
            os.makedirs(output_path, exist_ok=True)
            controlnet_output_name = output_path + "/controlnet2_{}.pt".format(self.batch)
            torch.onnx.export(jit_model, self.create_controlnet_input(), controlnet_output_name, opset_version=11,
                              verbose=False)
            # jit_model.save(controlnet_output_name)
            log.info(f" end controlnet convert")

    def run_vae(self):
        if self.unet_path is not None:
            log.info(f" start vae convert")
            self.export_vaencoder()
            self.export_vaedecoder()
            log.info(f" end vae convert")





