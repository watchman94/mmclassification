#python demo/image_embedding.py configs/resnet/resnet18_b32x8_custom.py weights/resnet18_b32x8/epoch_19.pth --folder /workspace/project/mtr/moco/data/sil/combine

#resnet 18
python demo/image_embedding.py configs/resnet/resnet18_8xb32_in1k.py weights/resnet18_b32x8/resnet18_8xb32_in1k_20210831-fbbb1da6.pth --folder /workspace/project/mtr/moco/data/sil/combine_lite --model-name resnet_18

#efficient b0, finetune
#python demo/image_embedding.py configs/efficientnet/efficientnet-b0_8xb32_custom.py weights/efficientnet-b0_8xb32/epoch_70.pth --folder /workspace/project/mtr/moco/data/sil/combine_lite --model-name efficientnet_b0

#efficient b0
#python demo/image_embedding.py configs/efficientnet/efficientnet-b0_8xb32_in1k.py weights/efficientnet-b0_8xb32/best_imagenet_b0.pth --folder /workspace/project/mtr/moco/data/sil/combine_lite --model-name efficientnet_b0

#efficient b3
#python demo/image_embedding.py configs/efficientnet/efficientnet-b3_8xb32-01norm_in1k.py weights/efficientnet-b3_8xb32/best_imagenet_b3.pth --folder /workspace/project/mtr/moco/data/sil/combine_lite --model-name efficientnet_b3
