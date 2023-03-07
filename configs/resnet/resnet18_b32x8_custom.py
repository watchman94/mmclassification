_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/custom_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

_deprecation_ = dict(
    expected='resnet18_8xb32_in1k.py',
    reference='https://github.com/open-mmlab/mmclassification/pull/508',
)

model = dict(
        head=dict(
            num_classes=13,
        )
)
