# dataset settings
dataset_type = 'CamVidDataset'
data_root = 'data/CamVid/'
crop_size = (960, 720)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(960, 720),
        ratio_range=(0.5, 2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

'''****************use train+val set to train*********************'''
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dict(
                    type='RepeatDataset',
                    times=1,
                    dataset=dict(
                        type=dataset_type,
                        data_root=data_root,
                        data_prefix=dict(
                            img_path='img_dir/train', seg_map_path='ann_dir/trainannot'),
                        pipeline=train_pipeline)
                    ),
                  dict(
                    type='RepeatDataset',
                    times=1,
                    dataset=dict(
                        type=dataset_type,
                        data_root=data_root,
                        data_prefix=dict(
                            img_path='img_dir/val', seg_map_path='ann_dir/valannot'),
                        pipeline=train_pipeline)
                    ),
                ]))


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dict(
                    type='RepeatDataset',
                    times=1,
                    dataset=dict(
                        type=dataset_type,
                        data_root=data_root,
                        data_prefix=dict(
                            img_path='img_dir/test', seg_map_path='ann_dir/testannot'),
                        pipeline=test_pipeline)
                    ),
                ]))


test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

