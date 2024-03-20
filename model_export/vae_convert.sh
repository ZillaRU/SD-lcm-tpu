decoder_shapes=("[1,4,16,48]" "[1,4,16,56]" "[1,4,16,64]" "[1,4,24,48]" "[1,4,24,56]" "[1,4,24,64]" "[1,4,32,48]" "[1,4,32,56]" "[1,4,32,64]" "[1,4,40,48]" "[1,4,40,56]" "[1,4,40,64]" "[1,4,48,16]" "[1,4,48,24]" "[1,4,48,32]" "[1,4,48,40]" "[1,4,48,48]" "[1,4,48,56]" "[1,4,48,64]" "[1,4,56,16]" "[1,4,56,24]" "[1,4,56,32]" "[1,4,56,40]" "[1,4,56,48]" "[1,4,56,56]" "[1,4,56,64]" "[1,4,64,16]" "[1,4,64,24]" "[1,4,64,32]" "[1,4,64,40]" "[1,4,64,48]" "[1,4,64,56]" "[1,4,64,64]" "[1,4,64,72]" "[1,4,64,80]" "[1,4,64,88]" "[1,4,64,96]" "[1,4,64,104]" "[1,4,64,112]" "[1,4,72,64]" "[1,4,80,64]" "[1,4,88,64]" "[1,4,96,64]" "[1,4,96,96]" "[1,4,104,64]" "[1,4,112,64]")

step=0
for i in "${decoder_shapes[@]}"; do
    t="model_transform.py --model_name vae_decoder --input_shape $i --model_def vae_decoder.pt --mlir vae_decoder.mlir"
    t1="model_deploy.py --mlir vae_decoder.mlir --quantize F16 --chip bm1684x --model vae_decoder_1684x_F16_$step.bmodel"
    step=$((step+1))
    `$t`
    `$t1`
    rm -rf ./*mlir && rm -rf ./*json && rm -rf ./*npz && rm -rf ./*profile
done

decoder_model=`ls | grep bmodel | grep decoder | grep -v txt | grep -v profile`
command="tpu_model --combine "
for i in $decoder_model; do
    command="$command $i"
done
command="$command -o vae_decoder_multize.bmodel" 
echo `$command`


encoder_shapes=("[1,3,128,384]" "[1,3,128,448]" "[1,3,128,512]" "[1,3,192,384]" "[1,3,192,448]" "[1,3,192,512]" "[1,3,256,384]" "[1,3,256,448]" "[1,3,256,512]" "[1,3,320,384]" "[1,3,320,448]" "[1,3,320,512]" "[1,3,384,128]" "[1,3,384,192]" "[1,3,384,256]" "[1,3,384,320]" "[1,3,384,384]" "[1,3,384,448]" "[1,3,384,512]" "[1,3,448,128]" "[1,3,448,192]" "[1,3,448,256]" "[1,3,448,320]" "[1,3,448,384]" "[1,3,448,448]" "[1,3,448,512]" "[1,3,512,128]" "[1,3,512,192]" "[1,3,512,256]" "[1,3,512,320]" "[1,3,512,384]" "[1,3,512,448]" "[1,3,512,512]" "[1,3,512,576]" "[1,3,512,640]" "[1,3,512,704]" "[1,3,512,768]" "[1,3,512,832]" "[1,3,512,896]" "[1,3,576,512]" "[1,3,640,512]" "[1,3,704,512]" "[1,3,768,512]" "[1,3,768,768]" "[1,3,832,512]" "[1,3,896,512]")

step=0
for i in "${encoder_shapes[@]}"; do
    t="model_transform.py --model_name vae_encoder --input_shape $i --model_def vae_decoder.pt --mlir vae_encoder.mlir"
    t1="model_deploy.py --mlir vae_encoder.mlir --quantize BF16 --chip bm1684x --model vae_encoder_1684x_F16_$step.bmodel"
    step=$((step+1))
    `$t`
    `$t1`
    rm -rf ./*mlir && rm -rf ./*json && rm -rf ./*npz && rm -rf ./*profile
done

encoder_model=`ls | grep bmodel | grep encoder | grep -v txt | grep -v profile`

command="tpu_model --combine "
for i in $encoder_model; do
    command="$command $i"
done
command="$command -o vae_encoder_multize.bmodel" 
echo `$command`