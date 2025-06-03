graph = "coco_onlyhand"
work_dir = f"/home/laptq/laptq-fs26-shoplifting-detection/runs/ProtoGCN/fs26/v87--mnit_poselift_roboflow_satudora_veo3misc_awlrecord--r2.0-0xauto-1x1--no-normal-awlrecord--1s-15frames"

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

dataset_type = "PoseDataset"
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
        ann_file="/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert_pkl_STGCN_to_ProtoGCN/fs26/mnit_poselift_roboflow_satudora_veo3misc_awlrecord--r2.0-0xauto-1x1--no-normal-awlrecord--1s-15frames/train.pkl",
        pipeline=train_pipeline,
        split="train",
    ),
    val=dict(
        type=dataset_type,
        ann_file="/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert_pkl_STGCN_to_ProtoGCN/fs26/shoplift25min_satudoraR9--r1-0x1-1x3--1s-15frames/val.pkl",
        pipeline=val_pipeline,
        split="val",
    ),
    test=dict(
        type=dataset_type,
        ann_file="/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert_pkl_STGCN_to_ProtoGCN/fs26/shoplift25min_satudoraR9--r1-0x1-1x3--1s-15frames/val.pkl",
        pipeline=test_pipeline,
        split="val",
    ),
)

# setting: 4 GPU  64  0.1  ->  1 GPU  64/4=16  0.1/4=0.025
optimizer = dict(
    type="SGD", lr=6.25e-5, momentum=0.9, weight_decay=0.0005, nesterov=True
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="CosineAnnealing", min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=-1)
evaluation = dict(interval=1, metrics=["top_k_accuracy"])
log_config = dict(interval=250, hooks=[dict(type="TextLoggerHook")])
