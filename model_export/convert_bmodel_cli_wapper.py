from ConvertorBmodel import ConvertorBmodel
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape_lists", type=int, nargs='+', default=[512,512])
    parser.add_argument("-v", "--sdversion", type=str, default="sd15")
    parser.add_argument("-n", "--path", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-o", "--output_bmodel", type=str, default="")
    args = parser.parse_args()

    # print(type(args.shape_lists))
    # print(args.shape_lists)

    convertor = ConvertorBmodel(
        shape_lists=[args.shape_lists],
        version=args.sdversion,
        path=args.path,
        batch=args.batch,
        output_bmodel=args.output_bmodel
    )

    def cli_run():
        convertor.convert_sd15_unet()
        print()
        convertor.convert_sd15_controlnet()
        print()
        convertor.convert_sd15_text_encoder()
        print()
        convertor.convert_sd15_vae_encoder()
        print()
        convertor.convert_sd15_vae_decoder()
        print()
        convertor.move_bmodels_into_folder()

    cli_run()
    print("check the folder in {}".format(convertor.output_bmodel))