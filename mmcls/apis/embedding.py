# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose

def get_embedding(model, img):
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

    my_embedding = torch.zeros(1, 512, 1, 1)
    extraction_layer = model.neck.gap
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = extraction_layer.register_forward_hook(copy_data)

    # forward the model
    with torch.no_grad():
        model(return_loss=False, **data)
        
    h.remove()
    my_embedding = my_embedding.numpy()[0, :, 0, 0]
    return my_embedding

