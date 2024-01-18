# wget https://github.com/sophgo/tpu-perf/releases/download/v1.2.17/tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl 
# 判断操作系统是arm还是x86
if [ $(uname -m) = "x86_64" ]; then
    echo "Your current operating system is based on x86_64"
    echo "you need to install the x86_64 version of sophon"
    echo "check your package now ..."
    pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    python3 -c "import sophon.sail" || echo "check package failed" && exit 1
    echo "check package success"
else
    echo "Your current operating system is based on arm64"
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.to/sharing/LiM00jkBJ && pip3 install sophon_arm-0.0.0-py3-none-any.whl && rm -rf ./sophon_arm-0.0.0-py3-none-any.whl
    pip3 install tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl
fi

pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if [ ! -d "./models" ]; then
    mkdir models
fi
if [ ! -d "./models/basic" ]; then
    mkdir models/basic
fi
if [ ! -d "./models/controlnet" ]; then
    mkdir models/controlnet
fi
if [ ! -d "./models/basic/babes20" ]; then
    python3 -m dfn --url http://disk-sophgo-vip.quickconnect.to/sharing/QWKCQ7Pxa && unzip babes20.zip && rm -rf babes20.zip && mv babes20 models/basic/
fi

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.to/sharing/63dydmQ6q && mv canny_multize.bmodel models/controlnet/
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.to/sharing/86Rm1E7cl && mv tile_multize.bmodel models/controlnet/
