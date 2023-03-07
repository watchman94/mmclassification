## 1. train the model with SIL Image
-  run this command(train.sh):
    ```shell
    python tools/train.py configs/resnet/resnet18_b32x8_custom.py
    ```
-  use CustomDataset in resnet18_b32x8_custom.py(custom_bs32.py)
     > use /home/kevin/workspaces/mtr/data_process/mtr.ipynb to create sil dataset

-  redefine the num_classes in model
    ```python
    model = dict(
        head=dict(
            num_classes=13,
        )
    )   
    ```
***

## 2. image to vector
-  run this command(get_embedding.sh)
    ```shell
    python demo/image_embedding.py \
    configs/resnet/resnet18_b32x8_custom.py \
    work_dirs/esnet18_b32x8_custom/epoch_19.pth \
    --folder /workspace/project/mtr/moco/data/sil/combine_lite
    --model-name efficientnet_b0
    ```
    > put the all images in folder
    >
    > put the best model in work_dirs
    > 
    > this shell save the embedding to embedding.json
    >
    > model_name decide the embedding extraction layer and embedding shape 

## 3. get the similarity
- run this command(test_img_similarity.py)
    ```shell
    python demo/test_img_similarity.py embedding.json --loc_threshold=3
    ```
    > loc_threshold means we just compare the image between [loc - 3, loc + 3]





