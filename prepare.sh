pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -U dfss
if [ ! -d "./models" ]; then
    mkdir models
fi
if [ ! -d "./models/basic" ]; then
    mkdir models/basic
fi
if [ ! -d "./models/controlnet" ]; then
    mkdir models/controlnet
fi
if [ ! -d "./models/basic/awportrait" ]; then
    python3 -m dfss --url=open@sophgo.com:/aigc/awportrait_lcm_models.zip && unzip awportrait_lcm_models.zip && rm -rf awportrait_lcm_models.zip && mv awportrait_lcm_models awportrait && mv awportrait models/basic/
fi
