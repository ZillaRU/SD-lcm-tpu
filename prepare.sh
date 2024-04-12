pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -U dfss
if [ ! -d "./models" ]; then
    mkdir models
fi
if [ ! -d "./models/basic" ]; then
    mkdir models/basic
fi
if [ ! -d "./models/controlnet" ]; then
    mkdir models/controlnet && python3 -m dfss --url=open@sophgo.com:/aigc/sd/canny_multize.bmodel && mv canny_multize.bmodel models/controlnet 
fi
if [ ! -d "./models/basic/awportrait" ]; then
    python3 -m dfss --url=open@sophgo.com:/aigc/sd/awportrait.tgz && tar xzvf awportrait.tgz && rm awportrait.tgz && mv awportrait models/basic/
fi
