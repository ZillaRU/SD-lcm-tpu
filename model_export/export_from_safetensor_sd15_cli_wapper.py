import argparse
import time
from ConvertorPtOnnx import ConvertorPtOnnx
import logging
log = logging.getLogger(__name__)


def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unet_safetensors_path', type=str, default=None)
    parser.add_argument('-c', '--controlnet_path', type=str, default=None)  # should be safetensors
    parser.add_argument('-l', '--lora_path', type=str, default=None)
    parser.add_argument('-cm', '--controlnet_merge', type=bool, default=False,
                        help="merge unet into controlnet with the former: \n " +
                             "new_controlnet = controlnet_weight - sd_base_encoder_weight + cur_unet_encoder_weight")
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-v", "--version", type=str, default="sd15")
    parser.add_argument('-o', '--output_name', type=str,
                        default=f"./tmp/{get_time_str()}")  # output_name should starts with ./tmp/
    parser.add_argument('-debug', '--debug_log', type=bool, default=False)

    args = parser.parse_args()
    print(args)

    convertor = ConvertorPtOnnx(
        unet_path=args.unet_safetensors_path,
        controlnet_path=args.controlnet_path,
        lora_path=args.lora_path,
        merge=args.controlnet_merge,
        batch=args.batch,
        version=args.version,
        output_name=args.output_name,
        debug=args.debug_log
    )

    def cli_run():
        log.info(f"start convert")
        convertor.init_convertor()
        convertor.init_pipe()
        convertor.run_controlnet()
        convertor.run_unet()
        convertor.run_text_encoder()
        convertor.run_vae()
        log.info(f"all done")

    cli_run()
    print("check the folder in {}".format(convertor.output_name))

