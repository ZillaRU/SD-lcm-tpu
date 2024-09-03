import os
import subprocess
import logging

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

class ConvertorBmodel():
    def __init__(self, shape_lists, version, path, batch, output_bmodel):
        self.shape_lists = shape_lists
        self.version = version
        self.path = path
        self.batch = batch
        self.output_bmodel = output_bmodel
        self.build_sd15_vae_encoder_shape = lambda x: [[1, 3, x[0], x[1]]]
        self.build_sd15_vae_decoder_shape = lambda x: [[1, 4, x[0] // 8, x[1] // 8]]

        self.init_convertor()


    def init_convertor(self):
        if self.output_bmodel == "":
            self.output_bmodel = "./bmodel/" + self.path.split("/")[-1]
        assert self.version in ["sd15"], "only support sd15"

    def _os_system_log(self, cmd_str):
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

    def _os_system_(self, cmd: str, save_log: bool = False):
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
            self._os_system_log(cmd_str)

    def _os_system(self, cmd, save_log=False):
        if isinstance(cmd, list):
            self._os_system(" ".join(cmd), save_log)
            return
            # print(cmd)
        if cmd.startswith("pushd"):
            pushd(cmd.replace("pushd", "").strip())
            return
        if cmd.startswith("popd"):
            popd()
            return
        self._os_system_(cmd, save_log)

    def remove_tmp_file(self, kfiles):
        cmd = ["rm -rf", "`ls"]
        for f in kfiles:
            cmd.append("| grep -v -x " + f)
        cmd.append("`")
        self._os_system(cmd)

    def convert_sd15_text_encoder(self):
        text_encoder_path = f"{self.path}/text_encoder"
        if os.path.isfile(text_encoder_path + "/text_encoder.onnx"):
            self._os_system(["pushd " + text_encoder_path])
            cmd = [
                "model_transform.py --model_name sdv15_te --input_shape [[1,77]] --model_def text_encoder.onnx --mlir sd15_te.mlir"]
            self._os_system(cmd)
            cmd = ["model_deploy.py --mlir sd15_te.mlir --quantize F32 --chip bm1684x --model sdv15_text.bmodel"]
            self._os_system(cmd)
            self.remove_tmp_file(["text_encoder.onnx", "sdv15_text.bmodel"])
            self._os_system(["popd"])
        else:
            log.warning("text_encoder.onnx not found, do not convert...")
            return 0
        return 1

    def build_sd15_controlnet_shape(self, batch=1, shape=[512, 512]):
        batch = int(batch)
        img_size = (512, 512)
        controlnet_latent_model_input = [batch, 4, img_size[0] // 8, img_size[1] // 8]
        controlnet_prompt_embeds = [batch, 77, 768]
        image = [batch, 3, img_size[0], img_size[1]]
        t = [1]
        return [controlnet_latent_model_input, controlnet_prompt_embeds, image, t]

    def build_sd15_unet_with_controlnet_interface_shape(self, batch=1, shape=[512, 512]):
        batch = int(batch)
        img_size = shape
        unet_latent_model_input = [batch, 4, img_size[0] // 8, img_size[1] // 8]
        t = [1]
        unet_prompt_embeds = [batch, 77, 768]
        mid_block_additional_residual = [batch, 1280, img_size[0] // 64, img_size[1] // 64]
        down_block_additional_residuals = []
        down_block_additional_residuals.append([batch, 320, img_size[0] // 8, img_size[1] // 8])
        down_block_additional_residuals.append([batch, 320, img_size[0] // 8, img_size[1] // 8])
        down_block_additional_residuals.append([batch, 320, img_size[0] // 8, img_size[1] // 8])
        down_block_additional_residuals.append([batch, 320, img_size[0] // 16, img_size[1] // 16])
        down_block_additional_residuals.append([batch, 640, img_size[0] // 16, img_size[1] // 16])
        down_block_additional_residuals.append([batch, 640, img_size[0] // 16, img_size[1] // 16])
        down_block_additional_residuals.append([batch, 640, img_size[0] // 32, img_size[1] // 32])
        down_block_additional_residuals.append([batch, 1280, img_size[0] // 32, img_size[1] // 32])
        down_block_additional_residuals.append([batch, 1280, img_size[0] // 32, img_size[1] // 32])
        down_block_additional_residuals.append([batch, 1280, img_size[0] // 64, img_size[1] // 64])
        down_block_additional_residuals.append([batch, 1280, img_size[0] // 64, img_size[1] // 64])
        down_block_additional_residuals.append([batch, 1280, img_size[0] // 64, img_size[1] // 64])
        return [unet_latent_model_input, t, unet_prompt_embeds, mid_block_additional_residual,
                *down_block_additional_residuals]

    def combine_models(self, model_paths, output_path=""):
        # name: xxx_shape1_shape2.bmodel -> xxx_multisize.bmodel
        log.info("start combine models for converting multi shape or multi net models into one net")
        cmd = ["model_tool.py --combine"]
        for model in model_paths:
            cmd.append(model)
        cmd.append("-o " + output_path)
        self._os_system(cmd)
        log.info("end combine models for converting multi shape or multi net models into one net")

    def rename_models(self, model_path, output_path):
        log.info("start rename models")
        cmd = ["mv", f"{model_path}", f"{output_path}"]
        self._os_system(cmd)
        log.info("end rename models")

    def check_path(self, path):
        if not os.path.exists(path):
            return True
        return False

    def convert_sd15_controlnet(self):
        controlnet_path = f"{self.path}/controlnet"
        if self.check_path(controlnet_path): return
        log.info("start convert controlnet model")
        file = os.listdir(controlnet_path)[0]
        batch = file.split("_")[-1].split(".")[0]
        controlnet_path = f"{controlnet_path}/controlnet_{batch}.pt"
        if os.path.isfile(controlnet_path):
            self._os_system("pushd " + controlnet_path)
            keep_file = ["controlnet_" + batch + ".pt"]
            for shape in self.shape_lists:
                controlnet_shape = self.build_sd15_controlnet_shape(batch, shape)
                shape_str = "_".join(str(i) for i in shape)
                cmd = ["model_transform.py --model_name sdv15_cn --input_shape", str(controlnet_shape).replace(" ", ""),
                       f"--model_def controlnet_{batch}.pt --mlir sd15_cn_{shape_str}.mlir"]
                self._os_system(cmd)
                cmd = [
                    f"model_deploy.py --mlir sd15_cn_{shape_str}.mlir --quantize F16 --chip bm1684x --model sdv15_cn_{shape_str}.bmodel"]
                self._os_system(cmd)
                keep_file.append(f"sdv15_cn_{shape_str}.bmodel")
                cmd = self.remove_tmp_file(keep_file)
            if len(self.shape_lists) > 1:
                self.combine_models(keep_file[1:], "sdv15_cn_multisize.bmodel")
                self.remove_tmp_file([keep_file[0], "sdv15_cn_multisize.bmodel"])
            else:
                # rename the file
                self.rename_models(keep_file[1], "sdv15_cn_multisize.bmodel")
                pass
            self._os_system("popd")
        log.info("end convert controlnet model")
        pass

    def convert_sd15_unet(self):
        unet_path = f"{self.path}/unet"
        unet_folder = unet_path
        if self.check_path(unet_path): return
        file = os.listdir(unet_path)[0]
        batch = file.split("_")[-1].split(".")[0]
        # print("++++{} {}".format(batch, type(batch)))
        isfuse = False if "fuse" not in file else True
        unet_path = f"{unet_path}/unet_{batch}.pt" if not isfuse else f"{unet_path}/unet_fuse_{batch}.pt"
        unet_pt_name = "unet_fuse" if isfuse else "unet"
        unet_pt_name += f"_{batch}.pt"
        if os.path.isfile(unet_path):
            self._os_system("pushd " + unet_folder)
            keep_file = [unet_pt_name]
            for shape in self.shape_lists:
                unet_shape = self.build_sd15_unet_with_controlnet_interface_shape(batch, shape)
                shape_str = "_".join(str(i) for i in shape)
                cmd = [f"model_transform.py --model_name sdv15_unet_{'fuse' if isfuse else 'no_fuse'} --input_shape",
                       str(unet_shape).replace(" ", ""),
                       f"--model_def {unet_pt_name} --mlir sd15_unet_{shape_str}.mlir"]
                self._os_system(cmd)
                cmd = [
                    f"model_deploy.py --mlir sd15_unet_{shape_str}.mlir --quantize F16 --chip bm1684x --model sdv15_unet_{shape_str}.bmodel"]
                self._os_system(cmd)
                keep_file.append(f"sdv15_unet_{shape_str}.bmodel")
                cmd = self.remove_tmp_file(keep_file)
            if len(self.shape_lists) > 1:
                self.combine_models(keep_file[1:], "sdv15_unet_multisize.bmodel")
                self.remove_tmp_file([keep_file[0], "sdv15_unet_multisize.bmodel"])
            else:
                # rename the file
                self.rename_models(keep_file[1], "sdv15_unet_multisize.bmodel")
            self._os_system("popd")

    def convert_sd15_vae_encoder(self):
        vae_encoder_path = f"{self.path}/vae_encoder"
        if os.path.isfile(vae_encoder_path + "/vae_encoder.pt"):
            # multi shape
            self._os_system("pushd " + vae_encoder_path)
            keep_file = ["vae_encoder.pt"]
            for shape in self.shape_lists:
                vae_encoder_shape = self.build_sd15_vae_encoder_shape(shape)
                shape_str = "_".join(str(i) for i in shape)
                cmd = ["model_transform.py --model_name sdv15_ve --input_shape",
                       str(vae_encoder_shape).replace(" ", ""),
                       f"--model_def vae_encoder.pt --mlir sd15_ve_{shape_str}.mlir"]
                self._os_system(cmd)
                cmd = [
                    f"model_deploy.py --mlir sd15_ve_{shape_str}.mlir --quantize F16 --chip bm1684x --model sdv15_ve_{shape_str}.bmodel"]
                self._os_system(cmd)
                keep_file.append(f"sdv15_ve_{shape_str}.bmodel")
                cmd = self.remove_tmp_file(keep_file)
            if len(self.shape_lists) > 1:
                self.combine_models(keep_file[1:], "sdv15_ve_multisize.bmodel")
                self.remove_tmp_file([keep_file[0], "sdv15_ve_multisize.bmodel"])
            else:
                # rename the file
                self.rename_models(keep_file[1], "sdv15_ve_multisize.bmodel")
            self._os_system("popd")
        pass

    def convert_sd15_vae_decoder(self):
        vae_decoder_path = f"{self.path}/vae_decoder"
        vae_decoder_folder = vae_decoder_path
        if self.check_path(vae_decoder_path): return
        log.info("start convert vae_decoder model")
        if os.path.isfile(vae_decoder_path + "/vae_decoder.pt"):
            # multi shape
            self._os_system("pushd " + vae_decoder_folder)
            keep_file = ["vae_decoder.pt"]
            for shape in self.shape_lists:
                vae_decoder_shape = self.build_sd15_vae_decoder_shape(shape)
                shape_str = "_".join(str(i) for i in shape)
                cmd = ["model_transform.py --model_name sdv15_vd --input_shape",
                       str(vae_decoder_shape).replace(" ", ""),
                       f"--model_def vae_decoder.pt --mlir sd15_vd_{shape_str}.mlir"]
                self._os_system(cmd)
                cmd = [
                    f"model_deploy.py --mlir sd15_vd_{shape_str}.mlir --quantize BF16 --chip bm1684x --model sdv15_vd_{shape_str}.bmodel"]
                self._os_system(cmd)
                keep_file.append(f"sdv15_vd_{shape_str}.bmodel")
                cmd = self.remove_tmp_file(keep_file)
            if len(self.shape_lists) > 1:
                self.combine_models(keep_file[1:], "sdv15_vd_multisize.bmodel")
                self.remove_tmp_file([keep_file[0], "sdv15_vd_multisize.bmodel"])
            else:
                # rename the file
                self.rename_models(keep_file[1], "sdv15_vd_multisize.bmodel")
            self._os_system("popd")
        pass

    def move_bmodels_into_folder(self):
        log.info("start copy bmodels into certain folder")
        os.makedirs(self.output_bmodel, exist_ok=True)
        for model in os.listdir(self.path):
            cur_model_path = os.path.join(self.path, model)
            for bmodel in os.listdir(cur_model_path):
                if ".bmodel" in bmodel:
                    cur_bmodel_path = os.path.join(cur_model_path, bmodel)
                    self._os_system(["cp", cur_bmodel_path, self.output_bmodel])
        log.info("end copy bmodels into certain folder")



