# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import mmcv
from mmcls.apis import get_embedding, init_model_new
import json

def get_all_embedding(model, folder, extraction_layer, shape):
    import os

    embedding_dic = {}
    for f in os.listdir(folder):
        if f.endswith("png") or f.endswith("jpg") or f.endswith("jpeg"):
            embedding = get_embedding(model, os.path.join(folder, f), extraction_layer, shape)
            embedding = embedding.tolist()
            embedding_dic[f] = embedding

    return embedding_dic

def main():
    parser = ArgumentParser()
    
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--img', required=False, help='Image file')
    parser.add_argument('--folder', required=False, help='Image directory')
    parser.add_argument('--model-name', required=True, help='model type, decide the extraction layer and embedding shape')
    args = parser.parse_args()

    if args.img == None and args.folder == None:
        print("please provide img or img's folder")
        return
    
    # build the model from a config file and a checkpoint file
    model, extraction_layer, shape = init_model_new(args.config, args.model_name, args.checkpoint, device=args.device)
    # test a single image
    
    if args.img: 
        result = get_embedding(model, args.img, extraction_layer, shape)
        print(result.shape)
        print(result)
        return

    if args.folder:
        embedding_dic = get_all_embedding(model, args.folder, extraction_layer, shape)
        json_object = json.dumps(embedding_dic, indent=4) 
        with open('embedding.json', 'w') as outfile:
            outfile.write(json_object)


if __name__ == '__main__':
    main()
