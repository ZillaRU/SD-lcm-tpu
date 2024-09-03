import shutil
import os
import subprocess
import logging
import argparse

directory_stack = []

def pushd(path):
    directory_stack.append(os.getcwd())
    os.chdir(path)

def popd():
    if not directory_stack:
        print("Directory stack is empty")
        return
    prev_dir = directory_stack.pop()
    os.chdir(prev_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shape_lists", type=int, nargs='+', default="512 512 768 768")
parser.add_argument("-v", "--sdversion", type=str, default="sd15")
parser.add_argument("-n", "--path", type=str)
parser.add_argument("-b", "--batch", type=int, default=1)
parser.add_argument("-o", "--output_bmodel", type=str, default="")
args = parser.parse_args()
args.shape_lists = [ [args.shape_lists[i], args.shape_lists[i+1]] for i in range(0, len(args.shape_lists), 2)]
if args.output_bmodel == "":
    args.output_bmodel = "./bmodel/" + args.path.split("/")[-1]
print(args)
assert args.sdversion in ["sd15"] , "only support sd15"
path = args.path

def _os_system_log(cmd_str):
    log.info("[Running]: %s", cmd_str)
    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    while True:
        output = process.stdout.readline().strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output)

    process.wait()
    ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))


def _os_system_(cmd: str, save_log: bool = False):
    # cmd_str = ""
    # for s in cmd:
    #     cmd_str += str(s) + " "
    cmd_str = cmd
    if not save_log:
        log.info("[Running]: %s", cmd_str)
        ret = os.system(cmd_str)
        if ret == 0:
            log.info("[Success]: %s", cmd_str)
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))
    else:
        _os_system_log(cmd_str)

def _os_system(cmd: list|str, save_log: bool = False):
    if isinstance(cmd, list):
        _os_system(" ".join(cmd), save_log)
        return 
    # print(cmd)
    if cmd.startswith("pushd") :
        pushd(cmd.replace("pushd","").strip())
        return
    if cmd.startswith("popd"):
        popd()
        return
    _os_system_(cmd, save_log)

def remove_tmp_file(kfiles):
    cmd = ["rm -rf", "`ls"]
    for f in kfiles:
        cmd.append("| grep -v -x "+f)
    cmd.append("`")
    _os_system(cmd)

def convert_sd15_text_encoder():
    text_encoder_path = f"{path}/text_encoder"
    if os.path.isfile(text_encoder_path+"/text_encoder.onnx"):
        _os_system(["pushd "+text_encoder_path])
        cmd = ["model_transform.py --model_name sdv15_te --input_shape [[1,77]] --model_def text_encoder.onnx --mlir sd15_te.mlir"]
        _os_system(cmd)
        cmd = ["model_deploy.py --mlir sd15_te.mlir --quantize F32 --chip bm1684x --model sdv15_text.bmodel"]
        _os_system(cmd)
        remove_tmp_file(["text_encoder.onnx","sdv15_text.bmodel"])
        _os_system(["popd"])
    else:
        log.warning("text_encoder.onnx not found, do not convert...")
        return 0
    return 1

def build_sd15_controlnet_shape(batch=1,shape=[512,512]):
    batch = int(batch)
    img_size = (512, 512)
    controlnet_latent_model_input = [batch, 4, img_size[0]//8, img_size[1]//8]
    controlnet_prompt_embeds = [batch, 77, 768]
    image = [batch, 3, img_size[0], img_size[1]]
    t = [1]
    return [controlnet_latent_model_input, controlnet_prompt_embeds, image, t]

def build_sd15_unet_with_controlnet_interface_shape(batch=1,shape=[512,512]):
    batch = int(batch)
    img_size = shape
    unet_latent_model_input = [batch, 4, img_size[0]//8, img_size[1]//8]
    t = [1]
    unet_prompt_embeds = [batch, 77, 768]
    mid_block_additional_residual = [batch, 1280, img_size[0]//64, img_size[1]//64]
    down_block_additional_residuals = []
    down_block_additional_residuals.append([batch, 320, img_size[0]//8, img_size[1]//8])
    down_block_additional_residuals.append([batch, 320, img_size[0]//8, img_size[1]//8])
    down_block_additional_residuals.append([batch, 320, img_size[0]//8, img_size[1]//8])
    down_block_additional_residuals.append([batch, 320, img_size[0]//16, img_size[1]//16])
    down_block_additional_residuals.append([batch, 640, img_size[0]//16, img_size[1]//16])
    down_block_additional_residuals.append([batch, 640, img_size[0]//16, img_size[1]//16])
    down_block_additional_residuals.append([batch, 640, img_size[0]//32, img_size[1]//32])
    down_block_additional_residuals.append([batch, 1280, img_size[0]//32, img_size[1]//32])
    down_block_additional_residuals.append([batch, 1280, img_size[0]//32, img_size[1]//32])
    down_block_additional_residuals.append([batch, 1280, img_size[0]//64, img_size[1]//64])
    down_block_additional_residuals.append([batch, 1280, img_size[0]//64, img_size[1]//64])
    down_block_additional_residuals.append([batch, 1280, img_size[0]//64, img_size[1]//64])
    return [unet_latent_model_input, t, unet_prompt_embeds, mid_block_additional_residual, *down_block_additional_residuals]

def combine_models(model_paths,output_path=""):
    # name: xxx_shape1_shape2.bmodel -> xxx_multisize.bmodel
    log.info("start combine models for converting multi shape or multi net models into one net")
    cmd = ["model_tool.py --combine"]
    for model in model_paths:
        cmd.append(model)
    cmd.append("-o "+output_path)
    _os_system(cmd)
    log.info("end combine models for converting multi shape or multi net models into one net")

def rename_models(model_path, output_path):
    log.info("start rename models")
    cmd = ["mv", f"{model_path}", f"{output_path}"]
    _os_system(cmd)
    log.info("end rename models")

def check_path(path):
    if not os.path.exists(path):
        return True
    return False

def convert_sd15_controlnet():
    controlnet_path = f"{path}/controlnet"
    if check_path(controlnet_path): return 
    log.info("start convert controlnet model")
    file = os.listdir(controlnet_path)[0]
    batch = file.split("_")[-1].split(".")[0]
    controlnet_path = f"{controlnet_path}/controlnet_{batch}.pt"
    if os.path.isfile(controlnet_path):
        _os_system("pushd "+controlnet_path)
        keep_file = ["controlnet_"+batch+".pt"]
        for shape in args.shape_lists:
            controlnet_shape = build_sd15_controlnet_shape(batch,shape)
            shape_str= "_".join(str(i) for i in shape)
            cmd = ["model_transform.py --model_name sdv15_cn --input_shape",str(controlnet_shape).replace(" ",""),f"--model_def controlnet_{batch}.pt --mlir sd15_cn_{shape_str}.mlir"]
            _os_system(cmd)
            cmd = [f"model_deploy.py --mlir sd15_cn_{shape_str}.mlir --quantize F16 --chip bm1684x --model sdv15_cn_{shape_str}.bmodel"]
            _os_system(cmd)
            keep_file.append( f"sdv15_cn_{shape_str}.bmodel" )
            cmd = remove_tmp_file(keep_file)
        if len(args.shape_lists) > 1:
            combine_models(keep_file[1:], "sdv15_cn_multisize.bmodel")
            remove_tmp_file([keep_file[0], "sdv15_cn_multisize.bmodel"])
        else:
            # rename the file
            rename_models(keep_file[1], "sdv15_cn_multisize.bmodel")
            pass
        _os_system("popd")
    log.info("end convert controlnet model")
    pass

def convert_sd15_unet():
    unet_path = f"{path}/unet"
    unet_folder = unet_path
    if check_path(unet_path): return
    file = os.listdir(unet_path)[0]
    batch = file.split("_")[-1].split(".")[0]
    isfuse = False if "fuse" not in file else True
    unet_path = f"{unet_path}/unet_{batch}.pt" if not isfuse else f"{unet_path}/unet_fuse_{batch}.pt"
    unet_pt_name =  "unet_fuse" if isfuse else "unet"
    unet_pt_name += f"_{batch}.pt"
    if os.path.isfile(unet_path):
        _os_system("pushd "+unet_folder)
        keep_file = [unet_pt_name]
        for shape in args.shape_lists:
            unet_shape = build_sd15_unet_with_controlnet_interface_shape(batch,shape)
            shape_str= "_".join(str(i) for i in shape)
            cmd = [f"model_transform.py --model_name sdv15_unet_{'fuse' if isfuse else 'no_fuse'} --input_shape",str(unet_shape).replace(" ",""),f"--model_def {unet_pt_name} --mlir sd15_unet_{shape_str}.mlir"]
            _os_system(cmd)
            cmd = [f"model_deploy.py --mlir sd15_unet_{shape_str}.mlir --quantize F16 --chip bm1684x --model sdv15_unet_{shape_str}.bmodel"]
            _os_system(cmd)
            keep_file.append( f"sdv15_unet_{shape_str}.bmodel" )
            cmd = remove_tmp_file(keep_file)
        if len(args.shape_lists) > 1:
            combine_models(keep_file[1:], "sdv15_unet_multisize.bmodel")
            remove_tmp_file([keep_file[0], "sdv15_unet_multisize.bmodel"])
        else:
            # rename the file
            rename_models(keep_file[1], "sdv15_unet_multisize.bmodel")
        _os_system("popd")

build_sd15_vae_encoder_shape = lambda x: [[1,3,x[0],x[1]]]
build_sd15_vae_decoder_shape = lambda x: [[1,4,x[0]//8,x[1]//8]]
def convert_sd15_vae_encoder():
    vae_encoder_path = f"{path}/vae_encoder"
    if os.path.isfile(vae_encoder_path+"/vae_encoder.pt"):
        # multi shape
        _os_system("pushd "+vae_encoder_path)
        keep_file = ["vae_encoder.pt"]
        for shape in args.shape_lists:
            vae_encoder_shape = build_sd15_vae_encoder_shape(shape)
            shape_str= "_".join(str(i) for i in shape)
            cmd = ["model_transform.py --model_name sdv15_ve --input_shape",str(vae_encoder_shape).replace(" ",""),f"--model_def vae_encoder.pt --mlir sd15_ve_{shape_str}.mlir"]
            _os_system(cmd)
            cmd = [f"model_deploy.py --mlir sd15_ve_{shape_str}.mlir --quantize F16 --chip bm1684x --model sdv15_ve_{shape_str}.bmodel"]
            _os_system(cmd)
            keep_file.append( f"sdv15_ve_{shape_str}.bmodel" )
            cmd = remove_tmp_file(keep_file)
        if len(args.shape_lists) > 1:
            combine_models(keep_file[1:], "sdv15_ve_multisize.bmodel")
            remove_tmp_file([keep_file[0], "sdv15_ve_multisize.bmodel"])
        else:
            # rename the file
            rename_models(keep_file[1], "sdv15_ve_multisize.bmodel")
        _os_system("popd")
    pass

def convert_sd15_vae_decoder():
    vae_decoder_path = f"{path}/vae_decoder"
    vae_decoder_folder = vae_decoder_path
    if check_path(vae_decoder_path): return
    log.info("start convert vae_decoder model")
    if os.path.isfile(vae_decoder_path+"/vae_decoder.pt"):
        # multi shape
        _os_system("pushd "+vae_decoder_folder)
        keep_file = ["vae_decoder.pt"]
        for shape in args.shape_lists:
            vae_decoder_shape = build_sd15_vae_decoder_shape(shape)
            shape_str= "_".join(str(i) for i in shape)
            cmd = ["model_transform.py --model_name sdv15_vd --input_shape",str(vae_decoder_shape).replace(" ",""),f"--model_def vae_decoder.pt --mlir sd15_vd_{shape_str}.mlir"]
            _os_system(cmd)
            cmd = [f"model_deploy.py --mlir sd15_vd_{shape_str}.mlir --quantize BF16 --chip bm1684x --model sdv15_vd_{shape_str}.bmodel"]
            _os_system(cmd)
            keep_file.append( f"sdv15_vd_{shape_str}.bmodel" )
            cmd = remove_tmp_file(keep_file)
        if len(args.shape_lists) > 1:
            combine_models(keep_file[1:], "sdv15_vd_multisize.bmodel")
            remove_tmp_file([keep_file[0], "sdv15_vd_multisize.bmodel"])
        else:
            # rename the file
            rename_models(keep_file[1], "sdv15_vd_multisize.bmodel")
        _os_system("popd")
    pass

def move_bmodels_into_folder():
    log.info("start copy bmodels into certain folder")
    os.makedirs(args.output_bmodel, exist_ok=True)
    for model in os.listdir(path):
        cur_model_path = os.path.join(path, model)
        for bmodel in os.listdir(cur_model_path):
            if ".bmodel" in bmodel:
                cur_bmodel_path = os.path.join(cur_model_path, bmodel)
                _os_system(["cp", cur_bmodel_path, args.output_bmodel])
    log.info("end copy bmodels into certain folder")

convert_sd15_unet()
print()
convert_sd15_controlnet()
print()
convert_sd15_text_encoder()
print()
convert_sd15_vae_encoder()
print()
convert_sd15_vae_decoder()
print()
move_bmodels_into_folder()
