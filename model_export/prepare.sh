#!/bin/bash -ex
# version checker 

# file checker 

if [ ! -e ~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14 ]; then 
    echo "you need to download models--openai--clip-vit-large-patch14"s
    python3 -m dfss --url=open@sophgo.com:/aigc/models--openai--clip-vit-large-patch14.zip
    unzip models--openai--clip-vit-large-patch14.zip
    mkdir -p ~/.cache/huggingface/hub/
    mv models--openai--clip-vit-large-patch14 ~/.cache/huggingface/hub/
    rm -rf ./models--openai--clip-vit-large-patch14.zip
fi

