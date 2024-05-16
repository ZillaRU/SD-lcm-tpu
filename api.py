# -*- coding: UTF-8 -*- 
from io import BytesIO
import io
from flask import Flask, render_template, request, send_file, g, jsonify, send_from_directory
import argparse
import os
import random
import cv2
import base64
from PIL import Image, ImageEnhance
import numpy as np
# engine
from sd import StableDiffusionPipeline
app = Flask(__name__)

TEST=False
DEVICE_ID=os.environ.get('DEVICE_ID', 0)
BASENAME = os.environ.get('BASENAME', 'awportraitv14')
CONTROLNET = os.environ.get('CONTROLNET', '')
RETURN_BASE64 = bool(int(os.environ.get('RETURN_BASE64', 1)))

SHAPES=[[512,512],[640,960],[960,640],[704,896],[896,704],[576,1024],[1024,576]]

def hanle_seed(seed):
    if seed == -1:
        seed = random.randint(0, 2 ** 31 - 1)
    return seed

def handle_base64_image(controlnet_image):
    # 目前只支持一个controlnet_image, 不可以是list
    if isinstance(controlnet_image, list):
        controlnet_image = controlnet_image[0]
    if controlnet_image.startswith("data:image"):
        controlnet_image = controlnet_image.split(",")[1]
        
    return controlnet_image

def handle_output_base64_image(image_base64):
    if not RETURN_BASE64:
        return image_base64
    if not image_base64.startswith("data:image"):
        image_base64 = "data:image/jpeg;base64," + image_base64
    return image_base64

def get_shape_by_ratio(width, height):
    ratio_shape = {
        1:[512,512],
        2/3:[640,960],
        3/2:[960,640],
        4/3:[704,896],
        3/4:[896,704],
        9/16:[576,1024],
        16/9:[1024,576],
    }
    ratio = width/height
    # 这个ratio找到最接近的ratio_shape
    ratio_shape_list = list(ratio_shape.keys())
    ratio_shape_list.sort(key=lambda x:abs(x-ratio))
    nshape = ratio_shape[ratio_shape_list[0]]
    print(nshape)
    return nshape

@app.before_first_request
def load_model():
    pipeline = StableDiffusionPipeline(
        basic_model=BASENAME,
        controlnet_name=CONTROLNET,
        scheduler="LCM")
    app.config['pipeline'] = pipeline
    print("register pipeline to app object.")
    print('pipeline is in app.config:', 'pipeline' in app.config)

@app.route('/')
def home():
    return "Welcome to SD-LCM-tpu"

@app.route('/txt2img', methods=['POST'])
def process_data():
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    prompt = data.get('prompt')
    negative_prompt = None #data.get('negative_prompt')
    num_inference_steps = 4 # int(data.get('steps'))
    guidance_scale = 0 # int(data.get('cfg_scale', 0))
    strength = float(data.get('denoising_strength', 0.7))
    sampler_index = "LCM" # data.get('sampler_index', "LCM")
    
    seed = int(data.get('seed'))
    if seed == -1:
        seed = random.randint(0, 2 ** 31 - 1)
    # =========== #
    # s_churn = int(data.get('s_churn',0))
    # s_noise = int(data.get('s_noise',1))
    # s_tmax = data.get('s_tmax')
    # s_tmin = data.get('s_tmin')
    # =========== #
    # n_iter = int(data.get('n_iter',1))
    subseed = int(data.get('subseed'))# 不可以为-1
    subseed_strength = float(data.get('subseed_strength'))
    seed_resize_from_h = data.get('seed_resize_from_h',1)
    seed_resize_from_w = data.get('seed_resize_from_w',1)
    # ========== #
    # firstphase_height = data.get('firstphase_height', 0) 
    # firstphase_width = data.get('firstphase_width', 0)   
    # ========== #
    n_iter = int(data.get('n_iter', 1))
    width = int(data.get('width', 512))
    height = int(data.get('height', 512))
    
    nwidth, nheight = get_shape_by_ratio(width, height)

    # override_settings = data.get('override_settings',{})
    # restore_faces = bool(data.get('restore_faces', False))
    # data 是否包含 args的参数 
    controlnet_image = None
    controlnet_name  = None
    flag = True
    init_image = None
    mask = None
    controlnet_args = {}
    if 'alwayson_scripts' in data:
        if "controlnet" in data['alwayson_scripts']:
            if "args" in data['alwayson_scripts']['controlnet']:
                controlnet_args = data['alwayson_scripts']['controlnet']['args'][0]
                if "enabled" in data['alwayson_scripts']['controlnet']['args'][0]:
                    if data['alwayson_scripts']['controlnet']['args'][0]['enabled']==False:
                        controlnet_name = None
                        controlnet_image= None
                        flag = False
                    else:
                        controlnet_name = data['alwayson_scripts']['controlnet']['args'][0]['module'] # must be hed and canny 
                        flag = True
                else:
                    flag = False
                    controlnet_name = None
                    controlnet_image= None
                if len(data['alwayson_scripts']['controlnet']['args']) ==1  and flag:
                    args_info = data['alwayson_scripts']['controlnet']['args'][0]
                    # import pdb;pdb.set_trace()
                    if 'image' in args_info:
                        controlnet_image = data['alwayson_scripts']['controlnet']['args'][0]['image']

                        if controlnet_image is not None and controlnet_image != "":
                            
                            controlnet_image = handle_base64_image(controlnet_image)
                            controlnet_image = base64.b64decode(controlnet_image)
                            controlnet_image = Image.open(io.BytesIO(controlnet_image))
                            # controlnet_image = np.array(controlnet_image)
                        else:
                            if init_image is not None:
                                controlnet_image = init_image
                            else:
                                controlnet_image = None
                else:
                    controlnet_image = None
                    controlnet_name = None

    init_image = None
    mask = None    
    with app.app_context():
        pipeline = app.config['pipeline']  # 获取 pipeline 变量
        pipeline.set_height_width(nheight, nwidth)
        try:
            pipeline.scheduler = sampler_index
            img_pil = pipeline(
                prompt=prompt,
                negative_prompt=None, #negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = None,# controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                scheduler=sampler_index,
            )
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            print(e)
            print("error")

    # img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if nwidth != width or nheight != height:
        img_pil = img_pil.resize((width, height))
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    ret_img_b64 = handle_output_base64_image(ret_img_b64)
    # 构建JSON响应
    response = jsonify({'code':0,'images': [ret_img_b64]})

    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route('/img2img', methods=['POST'])
def process_data_img():
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    prompt = data.get('prompt')
    negative_prompt = None # data.get('negative_prompt')
    num_inference_steps = 4 # int(data.get('steps', 4))
    guidance_scale = 0 # int(data.get('cfg_scale'))
    strength = float(data.get('denoising_strength', 0.4))
    seed = hanle_seed(int(data.get('seed')))
    
    sampler_index = "LCM" # data.get('sampler_index', "LCM")
    controlnet_image = None
    init_image = None
    mask = None
    init_image_b64 = data['init_images'][0]
    mask_image_b64 = data.get('mask') or None
    subseed = int(data.get('subseed'))# 不可以为-1
    subseed_strength = float(data.get('subseed_strength'))
    seed_resize_from_h = data.get('seed_resize_from_h',1)
    seed_resize_from_w = data.get('seed_resize_from_w',1)
    if init_image_b64:
        init_image_b64 = handle_base64_image(init_image_b64)
        init_image_bytes = BytesIO(base64.b64decode(init_image_b64))
        init_image = Image.open(init_image_bytes) # cv2.cvtColor(np.array(Image.open(init_image_bytes)), cv2.COLOR_RGB2BGR)
    if init_image_b64 and mask_image_b64:
        mask = BytesIO(base64.b64decode(mask_image_b64))
        mask[mask > 0] = 255
    else:
        mask = None

    controlnet_image = None
    controlnet_name  = None
    use_controlnet = True
    flag = True

    width = int(data.get('width', 512))
    height = int(data.get('height', 512))
    nwidth, nheight = get_shape_by_ratio(width, height)
    controlnet_args  = {}
    if 'alwayson_scripts' in data:
        if "controlnet" in data['alwayson_scripts']:
            if "args" in data['alwayson_scripts']['controlnet']:
                controlnet_args = data['alwayson_scripts']['controlnet']['args'][0]
                if "enabled" in data['alwayson_scripts']['controlnet']['args'][0]:
                    if data['alwayson_scripts']['controlnet']['args'][0]['enabled']==False:
                        use_controlnet = False
                        controlnet_name = None
                        controlnet_image= None
                        flag = False
                    else:
                        use_controlnet = True
                        controlnet_name = data['alwayson_scripts']['controlnet']['args'][0]['module'] # must be hed and canny 
                        flag = True
                else:
                    flag = False
                    controlnet_name = None
                    controlnet_image= None
                if len(data['alwayson_scripts']['controlnet']['args']) ==1  and flag:
                    args_info = data['alwayson_scripts']['controlnet']['args'][0]
                    if 'image' in args_info:
                        controlnet_image = data['alwayson_scripts']['controlnet']['args'][0]['image']
                        if controlnet_image is not None and controlnet_image != "":
                            controlnet_image = handle_base64_image(controlnet_image)
                            controlnet_image = base64.b64decode(controlnet_image)
                            controlnet_image = Image.open(io.BytesIO(controlnet_image))
                            controlnet_image = np.array(controlnet_image)
                        else:
                            if init_image is not None:
                                controlnet_image = init_image
                else:
                    controlnet_image = None
                    controlnet_name = None
                    

    with app.app_context():
        pipeline = app.config['pipeline']  # 获取 pipeline 变量
        pipeline.set_height_width(nheight, nwidth)
        try:
            pipeline.scheduler = sampler_index
            img_pil = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                use_controlnet = use_controlnet,
                scheduler=sampler_index
            )
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            print(e)
            print("error")

    # img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if nwidth != width or nheight != height:
        img_pil = img_pil.resize((width, height))
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    ret_img_b64 = handle_output_base64_image(ret_img_b64)
    # 构建JSON响应
    response = jsonify({'code':0,'images': [ret_img_b64]})

    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response



@app.route("/upscale", methods=['POST'])
def process_upscale():
    # =================================================#
    # 在upscale的时候 需要controlnetimg和initimg为同一张图
    # 但是为了传输方便 这里的controlnetimg可以为空 默认为原图
    # =================================================#
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt')
    num_inference_steps = int(data.get('steps'))
    guidance_scale = int(data.get('cfg_scale'))
    strength = float(data.get('denoising_strength'))
    seed = int(data.get('seed'))
    controlnet_image = None
    init_image = None
    mask = None
    init_image_b64 = data['init_images'][0]
    mask_image_b64 = data.get('mask') or None
    subseed = int(data.get('subseed'))# 不可以为-1
    subseed_strength = float(data.get('subseed_strength'))
    seed_resize_from_h = data.get('seed_resize_from_h',1)
    seed_resize_from_w = data.get('seed_resize_from_w',1)
    sampler_index = data.get('sampler_index', "Euler a")
    
    if init_image_b64:
        init_image_b64 = handle_base64_image(init_image_b64)
        init_image_bytes = BytesIO(base64.b64decode(init_image_b64))
        init_image = Image.open(init_image_bytes) # cv2.cvtColor(np.array(Image.open(init_image_bytes)), cv2.COLOR_RGB2BGR)
    if init_image_b64 and mask_image_b64:
        mask = BytesIO(base64.b64decode(mask_image_b64))
        mask[mask > 0] = 255
    else:
        mask = None
    controlnet_image = None
    controlnet_name  = None
    controlnet_args  = {}
    flag = True
    # upscale 参数处理 
    upscale_factor = int(data.get('upscale_factor', 2))# 必须大于0 且必须为整数
    target_width   = int(data.get('target_width', 1024))
    target_height  = int(data.get('target_height', 1024))
    # upscale和target必需传一个，两个都传的话以upscale_factor为准
    upscale_type   = data.get('upscale_type', 'LINEAR')# 必须大写 只有两种形式 LINEAR 和 CHESS
    tile_width     = int(data.get('tile_width', 512))# 目前tile大小规定为512 多tile的方式需要再测试
    tile_height    = int(data.get('tile_height', 512))# 目前tile大小规定为512 多tile的方式需要再测试
    mask_blur      = float(data.get('mask_blur', 8.0))
    padding        = int(data.get('padding', 32))
    upscaler       = data.get('upscaler', None)# placeholder 用于以后的超分模型
    seams_fix      = data.get('seams_fix', {})
    seams_fix_enable= bool(seams_fix.get('enable', False))# 目前没有开启缝隙修复
    

    if 'alwayson_scripts' in data:
        if "controlnet" in data['alwayson_scripts']:
            if "args" in data['alwayson_scripts']['controlnet']:
                controlnet_args = data['alwayson_scripts']['controlnet']['args'][0]
                if "enabled" in data['alwayson_scripts']['controlnet']['args'][0]:
                    if data['alwayson_scripts']['controlnet']['args'][0]['enabled']==False:
                        controlnet_name = None
                        controlnet_image= None
                        flag = False
                    else:
                        controlnet_name = data['alwayson_scripts']['controlnet']['args'][0]['module'] # must be hed and canny 
                        flag = True
                else:
                    flag = False
                    controlnet_name = None
                    controlnet_image= None
                if len(data['alwayson_scripts']['controlnet']['args']) ==1  and flag:
                    args_info = data['alwayson_scripts']['controlnet']['args'][0]
                    if 'image' in args_info:
                        controlnet_image = data['alwayson_scripts']['controlnet']['args'][0]['image']
                        # base64 to image
                        if controlnet_image is not None and controlnet_image != "":
                            controlnet_image = handle_base64_image(controlnet_image)
                            controlnet_image = base64.b64decode(controlnet_image)
                            controlnet_image = Image.open(io.BytesIO(controlnet_image))
                            # controlnet_image = np.array(controlnet_image)
                        else:
                            if init_image is not None:
                                controlnet_image = init_image
                else:
                    controlnet_image = None
                    controlnet_name = None
                    
    with app.app_context():
        pipeline = app.config['pipeline']  # 获取 pipeline 变量
        try:
            pipeline.scheduler = sampler_index
            image = pipeline.wrap_upscale(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                # upscale 参数
                upscale_factor = upscale_factor,
                target_width = target_width,
                target_height = target_height,
                upscale_type = upscale_type,
                mask_blur = mask_blur,
                tile_width = tile_width,
                tile_height = tile_height,
                padding   = padding,
                seams_fix_enable = seams_fix_enable,
                upscaler = upscaler,
                seams_fix = seams_fix,
                # upscale end
            )
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            print(e)
            print("error")

    # img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil = image
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    ret_img_b64 = handle_output_base64_image(ret_img_b64)
    # 构建JSON响应
    response = jsonify({'code':0,'images': [ret_img_b64]})
    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


if __name__ == "__main__":
    # engine setup
    app.run(debug=False, port=7019, host="0.0.0.0", threaded=False)
