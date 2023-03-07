# Copyright (c) OpenMMLab. All rights reserved.
import torch
import mmcv
import warnings
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier

def get_embedding(model, img, extraction_layer, shape):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    #my_embedding = torch.zeros(1, 512, 1, 1)
    #extraction_layer = model.neck.gap
    my_embedding = torch.zeros(shape[0], shape[1], shape[2], shape[3])
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = extraction_layer.register_forward_hook(copy_data)

    # forward the model
    with torch.no_grad():
        model(return_loss=False, **data)
        
    h.remove()
    my_embedding = my_embedding.numpy()[0, :, 0, 0]
    return my_embedding

def init_model_new(config, model_name, checkpoint=None, device='cuda:0', options=None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use imagenet by default.')
            model.CLASSES = ImageNet.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    #define extraction_layer, shape based on model_name
    if model_name == "resnet_18":
        extraction_layer = model.neck.gap
        shape = (1, 512, 1, 1)
    elif model_name == "efficientnet_b0":
        extraction_layer = model.neck.gap
        shape = (1, 1280, 1, 1)
    else:
        raise KeyError('Model %s was not found' % model_name)

    return model, extraction_layer, shape

