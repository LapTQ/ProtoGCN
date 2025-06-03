graph = "coco_onlyhand"
work_dir = f"/home/laptq/laptq-fs26-shoplifting-detection/runs/ProtoGCN/fs26/test"

model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="ProtoGCN",
        num_prototype=100,
        in_channels=2,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
        graph_cfg=dict(
            layout=graph, mode="random", num_filter=8, init_off=0.04, init_std=0.02
        ),
    ),
    cls_head=dict(
        type="SimpleHead",
        joint_cfg="coco_onlyhand",
        num_classes=2,
        in_channels=384,
        weight=0.2,
    ),
)

dataset_type = "PoseDatasetEvalFnormalEshoplift"
train_pipeline = [
    dict(type="NormalizeByMinMaxKeypoints"),
    dict(type="Normalize_01_to_neg11"),
    dict(type="SelectKeypoints", indexes=[5, 6, 7, 8, 9, 10]),
    dict(type="RandomRot", theta=0.2),
    dict(type="FormatGCNInput_v2"),
]
val_pipeline = [
    dict(type="NormalizeByMinMaxKeypoints"),
    dict(type="Normalize_01_to_neg11"),
    dict(type="SelectKeypoints", indexes=[5, 6, 7, 8, 9, 10]),
    dict(type="FormatGCNInput_v2"),
]
test_pipeline = [
    dict(type="NormalizeByMinMaxKeypoints"),
    dict(type="Normalize_01_to_neg11"),
    dict(type="SelectKeypoints", indexes=[5, 6, 7, 8, 9, 10]),
    dict(type="FormatGCNInput_v2"),
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file="/home/laptq/laptq-fs26-shoplifting-detection/outputs/autotrain_pipeline/fs26/no_satudora-500_max_entropy6/train_data.pkl",
        pipeline=train_pipeline,
        split="train",
    ),
    val=dict(
        type=dataset_type,
        ann_file="/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert_pkl_STGCN_to_ProtoGCN/fs26/tiny-val/val.pkl",
        pipeline=val_pipeline,
        split="val",
    ),
    test=dict(
        type=dataset_type,
        ann_file="/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert_pkl_STGCN_to_ProtoGCN/fs26/tiny-val/val.pkl",
        pipeline=test_pipeline,
        split="val",
    ),
)

# setting: 4 GPU  64  0.1  ->  1 GPU  64/4=16  0.1/4=0.025
optimizer = dict(
    type="SGD", lr=6.25e-3, momentum=0.9, weight_decay=0.0005, nesterov=True
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="CosineAnnealing", min_lr=6.25e-8, by_epoch=False, warmup='linear', warmup_iters=2, warmup_ratio=0.01, warmup_by_epoch=True)
total_epochs = 20
num_models = 2
checkpoint_config = dict(interval=-1, save_last=False)
evaluation = dict(
    interval=1, 
    metrics="F1_Fnormal_Eshoplift", 
    greater_keys=["F1_Fnormal_Eshoplift"],
    other_kwargs=dict(    
        enable__normalize_by_minmax=True,
        enable__rescale_to_neg1_pos1=True,
        layout="coco_onlyhand",
        num_prototype=100,
        in_channels=2,
        num_class=2,
        device_val=0,   # after mask CUDA_VISIBLE_DEVICES
        class_map=None
    )
)
log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook")])
enable_deterministic_hook = False
